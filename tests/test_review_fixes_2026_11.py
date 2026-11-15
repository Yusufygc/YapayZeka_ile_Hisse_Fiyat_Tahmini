# -*- coding: utf-8 -*-
"""Kod inceleme bulgulari (2026-11-16) icin regresyon testleri."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FeaturePipeline
from src.forecasting import workflows as wf
from src.forecasting.workflows import ForecastPointGenerator, ProductionTrainingWorkflow
from src.pipeline.data_services import model_feature_names


# --------------------------------------------------------------------------- #
#  Ortak yardimcilar                                                          #
# --------------------------------------------------------------------------- #
def _ohlcv(n: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2025-01-01", periods=n),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(10_000, 90_000, n).astype(float),
        }
    )


class _IdScaler:
    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = n_features

    def transform(self, values):
        return np.asarray(values, dtype=float)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float)


class _Rules:
    def bound_forecast_price(self, raw, prev):
        return float(raw), SimpleNamespace(lower_band=None, upper_band=None, price_tick=None)

    def next_trading_days(self, last, n):
        return list(pd.bdate_range(pd.Timestamp(last) + pd.Timedelta(days=1), periods=n))


class _Predictor:
    """Her adimda alternatif isaretli getiri: fiyat yatay kalmasin."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, model_name, model, context):
        self.calls += 1
        return 0.02 if self.calls % 2 else -0.015


def _artifact_style_context(monkeypatch):
    raw = _ohlcv()
    pipe = FeaturePipeline()
    engineered = pipe.engineer_features(raw)
    features = model_feature_names(pipe.feature_names)
    assert "NATR_14" in features and "SMA_7_rel" in features and "day_of_week" in features

    artifact = SimpleNamespace(
        model=object(),
        scaler_X=_IdScaler(len(features)),
        scaler_y=_IdScaler(1),
        metadata={"feature_names": features},
    )
    monkeypatch.setattr(wf, "load_forecast_artifact_package", lambda **kw: artifact)
    ctx = SimpleNamespace(
        project_root=".",
        rules=_Rules(),
        model_config=SimpleNamespace(model_settings={}),
        make_model_instance=lambda name, stage="final": object(),
        target_to_price=lambda target, prev, mode: prev * (1.0 + float(target)),
    )
    dm = SimpleNamespace(
        df=engineered,
        feature_names=features,
        data_cfg=SimpleNamespace(target_mode="return", feature_mode="stationary_features"),
    )
    training = ProductionTrainingWorkflow(ctx)
    _, context = training.train("Ridge Return", dm, selection={})
    return ctx, context


# --------------------------------------------------------------------------- #
#  1. feature_frame OHLV tasir; recursive teknik gostergeler donmaz           #
# --------------------------------------------------------------------------- #
def test_recursive_artifact_context_recomputes_technicals_without_warning(monkeypatch):
    ctx, context = _artifact_style_context(monkeypatch)
    for col in ("Open", "High", "Low", "Volume"):
        assert col in context["feature_frame"].columns

    gen = ForecastPointGenerator(ctx)
    seen: list[dict[str, float]] = []
    original = ForecastPointGenerator._refresh_latest_context

    def _spy(self, c, frame, model_name):
        row = frame.iloc[-1]
        seen.append({k: float(row[k]) for k in ("NATR_14", "SMA_7_rel", "RSI_14")})
        return original(self, c, frame, model_name)

    monkeypatch.setattr(ForecastPointGenerator, "_refresh_latest_context", _spy)
    gen.roll_forward_recursive(
        model_name="Ridge Return",
        model=object(),
        context=context,
        predictor=_Predictor(),
        horizon_days=5,
    )
    warns = context["forecast_warnings"]
    assert not [w for w in warns if w.startswith("recompute_failed")], warns
    assert len(seen) == 5
    for key in ("NATR_14", "SMA_7_rel", "RSI_14"):
        values = [s[key] for s in seen]
        assert len(set(np.round(values, 12))) == len(values), (key, values)


# --------------------------------------------------------------------------- #
#  2. day_of_week yeni satirin tarihinden gelir                               #
# --------------------------------------------------------------------------- #
def test_append_recursive_row_updates_day_of_week():
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-07", "2026-05-08"]),  # Per, Cum
            "Close": [100.0, 101.0],
            "day_of_week": [3.0, 4.0],
        }
    )
    out = ForecastPointGenerator._append_recursive_row(
        frame=frame,
        target_date=pd.Timestamp("2026-05-11"),  # Pazartesi
        bounded_close=102.0,
        previous_close=101.0,
    )
    assert out["day_of_week"].iloc[-1] == 0.0


# --------------------------------------------------------------------------- #
#  3. Kaldirilmis / bilinmeyen model secimi sessizce bos kume DONMEZ          #
# --------------------------------------------------------------------------- #
def test_normalize_candidate_models_rejects_removed_model():
    from src.pipeline.model_scope import normalize_candidate_models

    with pytest.raises(ValueError, match="LSTM"):
        normalize_candidate_models(["LSTM"])
    with pytest.raises(ValueError, match="LSTM"):
        normalize_candidate_models(["Ridge Return", "LSTM"])


def test_resolve_candidates_rejects_removed_and_unknown_models():
    from src.pipeline.model_scope import resolve_candidates

    with pytest.raises(ValueError) as excinfo:
        resolve_candidates(selected=["LSTM", "__Bogus__"])
    msg = str(excinfo.value)
    assert "LSTM" in msg and "__Bogus__" in msg


def test_normalize_candidate_models_still_accepts_benchmark_names():
    # --role benchmark gibi filtreler benchmark isimlerini selected'e koyar; hata degil.
    from src.pipeline.model_scope import normalize_candidate_models

    assert normalize_candidate_models(["XGBoost", "Naive Last Value"]) == {"XGBoost"}


def test_batch_cli_reports_removed_model_without_traceback(monkeypatch, capsys):
    from src.cli import batch

    monkeypatch.setattr("sys.argv", ["batch", "--stocks", "AAA", "--models", "LSTM", "--dry-run"])
    with pytest.raises(SystemExit) as excinfo:
        batch.main()
    assert excinfo.value.code == 2
    out = capsys.readouterr()
    assert "LSTM" in (out.out + out.err)
    assert "Traceback" not in (out.out + out.err)


def test_pipeline_job_with_removed_model_fails_with_message(tmp_path):
    from src.api.services import pipeline_jobs

    class _Tasks:
        def __init__(self):
            self.tasks = []

        def add_task(self, fn):
            self.tasks.append(fn)

    tasks = _Tasks()
    job_id = pipeline_jobs.start_pipeline_job(
        "AAA",
        pipeline_jobs.RunRequest(models=["LSTM"]),
        str(tmp_path),
        tasks,
    )
    for fn in tasks.tasks:
        fn()
    job = pipeline_jobs.get_job(job_id)
    assert job["status"] == "error"
    assert "LSTM" in job["error"]


# --------------------------------------------------------------------------- #
#  4. best_models siralamasi kaldirilmis modelleri disarida birakir           #
# --------------------------------------------------------------------------- #
def _log_prod(db, model_name, *, dir_acc, sharpe, run_id, rmse=2.0):
    db.log_experiment(
        stock_symbol="TUPRS",
        model_name=model_name,
        metrics={"MAE": rmse / 2, "RMSE": rmse, "Dir_Acc": dir_acc, "Sharpe": sharpe, "Trade_Count": 20},
        dataset_hash="hash1",
        validation_mode="final_holdout",
        is_production_candidate=True,
        run_id=run_id,
    )


@pytest.mark.parametrize("removed", ["LSTM", "Random Forest", "ElasticNet Return"])
def test_best_model_refresh_skips_removed_models(tmp_path, removed):
    from src.database.stock_model_db import StockModelDB

    path = str(tmp_path / "stock_models.db")
    db = StockModelDB(path)
    _log_prod(db, removed, dir_acc=70.0, sharpe=3.0, run_id="r1")
    _log_prod(db, "XGBoost", dir_acc=52.0, sharpe=0.1, run_id="r2", rmse=6.0)

    # Kaldirilmis modelin skoru acikca ustun: siralama onu seciyor olurdu.
    with db._connect() as conn:
        conn.execute("UPDATE experiments SET composite_score = 90 WHERE model_name = ?", (removed,))
        conn.execute("UPDATE experiments SET composite_score = 40 WHERE model_name = 'XGBoost'")
    db.ensure_schema(force=True)  # -> refresh_best_models_from_production_experiments
    assert db.get_best_model("TUPRS")["model_name"] == "XGBoost"


def test_best_model_refresh_keeps_removed_when_no_supported_row(tmp_path):
    from src.database.stock_model_db import StockModelDB

    path = str(tmp_path / "stock_models.db")
    db = StockModelDB(path)
    _log_prod(db, "LSTM", dir_acc=60.0, sharpe=1.0, run_id="r1")
    db.ensure_schema(force=True)
    assert db.get_best_model("TUPRS")["model_name"] == "LSTM"


# --------------------------------------------------------------------------- #
#  5/6. Refresh: once veri guncelle, sonra yapisal "retrain" sonucu           #
# --------------------------------------------------------------------------- #
def _refresh_service(tmp_path, monkeypatch):
    from src.api.services import data_refresh_service as mod

    calls: list[str] = []

    class _Update:
        def to_dict(self):
            return {"status": "updated"}

    def _fake_update(data_file, symbol, interactive=False):
        calls.append(symbol)
        return _Update()

    monkeypatch.setattr(mod.DataUpdater, "check_and_update", staticmethod(_fake_update))
    monkeypatch.setattr(mod, "ensure_bist_calendar", lambda *a, **k: None)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "TUPRS.csv").write_text("Date,Close\n2026-05-14,100\n2026-05-15,101\n", encoding="utf-8")
    service = mod.DataRefreshService(db=SimpleNamespace(db_path=None), project_root=str(tmp_path),
                                     start_background=False)
    return mod, service, calls


def test_refresh_removed_model_updates_data_first_then_reports_retrain(tmp_path, monkeypatch):
    mod, service, calls = _refresh_service(tmp_path, monkeypatch)
    with pytest.raises(mod.DataRefreshError) as exc:
        service.refresh_symbol(symbol="TUPRS", best_model={"model_name": "LSTM", "experiment_id": 1})
    assert calls == ["TUPRS"], "piyasa verisi guncellenmeden retrain hatasi verildi"
    assert exc.value.reason == "model_no_longer_supported"
    assert exc.value.payload["action"] == "retrain"
    assert exc.value.payload["data_update"] == {"status": "updated"}


def test_refresh_maps_outdated_artifact_features_to_retrain(tmp_path, monkeypatch):
    from src.forecasting.artifacts import ArtifactFeaturesOutdatedError

    mod, service, calls = _refresh_service(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "MacroPipeline", lambda **k: SimpleNamespace(get_macro_features=lambda **kw: None))
    service.db = SimpleNamespace(db_path="x.db", resolve_forecasts_from_csv=lambda *a, **k: 0)

    class _Runner:
        def __init__(self, **kw):
            pass

        def run_symbol(self, **kw):
            raise ArtifactFeaturesOutdatedError("XGBoost", "artifact feature mismatch: ...")

    import src.forecasting.runner as runner_mod

    monkeypatch.setattr(runner_mod, "ForecastRunner", _Runner)
    with pytest.raises(mod.DataRefreshError) as exc:
        service.refresh_symbol(symbol="TUPRS", best_model={"model_name": "XGBoost", "experiment_id": 1})
    assert exc.value.reason == "artifact_features_outdated"
    assert exc.value.payload["action"] == "retrain"
    assert exc.value.payload["model_name"] == "XGBoost"


# --------------------------------------------------------------------------- #
#  6. Eski ozellik listesi -> yapisal hata                                    #
# --------------------------------------------------------------------------- #
def test_validate_artifact_features_raises_structured_outdated_error():
    from src.forecasting.artifacts import ArtifactFeaturesOutdatedError, ForecastArtifactError

    with pytest.raises(ArtifactFeaturesOutdatedError) as exc:
        ProductionTrainingWorkflow._validate_artifact_features(
            model_name="XGBoost",
            features=["Return", "EMA_7_rel"],
            frame_columns=["Date", "Close", "Return"],
            current_features=["Return"],
            scaler_X=None,
        )
    assert isinstance(exc.value, ForecastArtifactError)
    assert exc.value.reason == "artifact_features_outdated"
    assert exc.value.as_payload()["action"] == "retrain"
    assert exc.value.model_name == "XGBoost"


# --------------------------------------------------------------------------- #
#  7. Gecelik cikis: terk edilen cagri varsa os._exit                         #
# --------------------------------------------------------------------------- #
def test_call_with_deadline_counts_abandoned_calls():
    import threading

    from src.data import market_providers as mp

    mp.reset_abandoned_calls()
    release = threading.Event()
    with pytest.raises(mp.ProviderTimeout):
        mp.call_with_deadline(lambda: release.wait(5.0), 0.05, "unit")
    release.set()
    assert mp.abandoned_call_count() == 1
    mp.reset_abandoned_calls()


def test_exit_after_run_forces_exit_only_when_calls_abandoned(monkeypatch):
    import logging

    from src.serving import nightly

    shutdowns: list[bool] = []
    monkeypatch.setattr(logging, "shutdown", lambda: shutdowns.append(True))

    assert nightly.should_force_exit(0) is False
    assert nightly.should_force_exit(2) is True

    # Terk edilen cagri yok -> normal SystemExit(rc), os._exit cagrilmaz.
    called: list[int] = []
    with pytest.raises(SystemExit) as excinfo:
        nightly.exit_after_run(3, exit_fn=called.append, abandoned_calls=0)
    assert excinfo.value.code == 3 and called == []

    # Terk edilen cagri var -> exit_fn(rc) cagrilir, SystemExit firlatilmaz.
    nightly.exit_after_run(5, exit_fn=called.append, abandoned_calls=1)
    assert called == [5] and shutdowns == [True]


# --------------------------------------------------------------------------- #
#  8. PowerShell: $ErrorActionPreference=Stop altinda Write-Error + exit      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["nightly_serving.ps1", "register_nightly_task.ps1"])
def test_ps1_error_paths_do_not_use_write_error_before_exit(name):
    import re
    from pathlib import Path

    raw = (Path(__file__).resolve().parents[1] / "scripts" / name).read_bytes()
    assert all(b < 128 for b in raw), "betik ASCII kalmali"
    text = raw.decode("ascii")
    # Write-Error, Stop tercihinde sonlandirici hata atar; `exit N` hic calismaz.
    assert not re.search(r"Write-Error[^\r\n]*\r?\n\s*exit\s+\d", text)


# --------------------------------------------------------------------------- #
#  10. market_momentum_60d piyasa geneli (makro takviminde, tarihe gore)      #
# --------------------------------------------------------------------------- #
def _momentum_stock(dates, seed):
    rng = np.random.default_rng(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    return pd.DataFrame({"Date": dates, "Close": close})


def test_market_momentum_is_identical_across_stocks_with_missing_days():
    n = 150
    dates = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(3)
    macro = pd.DataFrame(
        {"Date": dates, "BIST100_Return": rng.normal(0.0005, 0.012, n)}
    )
    full = _momentum_stock(dates, 1)
    gappy_dates = dates.delete([20, 45, 46, 80, 100, 101, 102])  # eksik islem gunleri
    gappy = _momentum_stock(gappy_dates, 2)

    fp = FeaturePipeline(lag_feature_count=0)
    m_full = fp._merge_macro(full, macro).set_index("Date")
    m_gappy = fp._merge_macro(gappy, macro).set_index("Date")

    common = m_gappy.index.intersection(m_full.index)
    a = m_full.loc[common, "market_momentum_60d"].dropna()
    b = m_gappy.loc[a.index, "market_momentum_60d"]
    assert len(a) > 30
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12)

    # Beklenen: BIST100 getirilerinin makro takvimindeki 60g kumulatifi.
    expected = np.expm1(np.log1p(macro.set_index("Date")["BIST100_Return"]).rolling(60).sum())
    np.testing.assert_allclose(
        a.to_numpy(), expected.loc[a.index].to_numpy(), rtol=0, atol=1e-12
    )
    # relative_to_market_60d = momentum_60d - market_momentum_60d korunur.
    rel = m_gappy["relative_to_market_60d"] - (
        m_gappy["momentum_60d"] - m_gappy["market_momentum_60d"]
    )
    assert rel.dropna().abs().max() < 1e-12


def test_recompute_momentum_keeps_market_history_and_refreshes_last_row():
    n = 150
    dates = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(5)
    macro = pd.DataFrame({"Date": dates, "BIST100_Return": rng.normal(0.0, 0.01, n)})
    gappy = _momentum_stock(dates.delete([30, 31, 70]), 4)
    fp = FeaturePipeline(lag_feature_count=0)
    merged = fp._merge_macro(gappy, macro).dropna().reset_index(drop=True)
    history = merged["market_momentum_60d"].copy()

    stale = merged.copy()
    stale.loc[stale.index[-1], "BIST100_Return"] = 0.0
    out = fp.recompute_momentum_features(stale)
    pd.testing.assert_series_equal(
        out["market_momentum_60d"].iloc[:-1], history.iloc[:-1], check_names=False
    )
    assert out["relative_to_market_60d"].iloc[-1] == pytest.approx(
        out["momentum_60d"].iloc[-1] - out["market_momentum_60d"].iloc[-1]
    )
