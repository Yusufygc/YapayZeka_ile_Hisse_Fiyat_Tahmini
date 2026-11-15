# -*- coding: utf-8 -*-
"""P4: src.serving.nightly tek modul + ince tools sarmalayicilari + Kol-B alan baglama."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import src.serving.nightly as nl
import src.serving.nightly_scoring as ns

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, "tools", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- CLI / wrappers
def test_module_help_works():
    out = subprocess.run([sys.executable, "-m", "src.serving.nightly", "--help"],
                         cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0
    for flag in ("--fallback-dir", "--quarantine-jumps", "--skip-data", "--skip-trading-gate",
                 "--model", "--liq-floor-tl", "--stale-days", "--limit", "--sleep-s"):
        assert flag in out.stdout


@pytest.mark.parametrize("name,attr", [("e2_nightly_pipeline", "main"),
                                       ("e2_faz5_nightly_scoring", "score_main")])
def test_tool_wrappers_delegate(name, attr):
    mod = _load_tool(name)
    assert getattr(mod, attr) is getattr(nl, attr)
    with open(os.path.join(_REPO, "tools", f"{name}.py"), encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) <= 10


def test_score_main_runs_scoring_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(nl, "run_scoring", lambda a: seen.update(fb=a.fallback_dir, qj=a.quarantine_jumps) or 0)
    monkeypatch.setattr(nl, "refresh_universe",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tazeleme yok")))
    monkeypatch.setattr(nl, "is_trading_day",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("kapi yok")))
    assert nl.score_main(["--fallback-dir", "a", "--fallback-dir", "b", "--no-quarantine-jumps"]) == 0
    assert seen == {"fb": ["a", "b"], "qj": False}


def test_ps1_uses_module_fallback_dir_and_no_hardcoded_python():
    with open(os.path.join(_REPO, "scripts", "nightly_serving.ps1"), encoding="utf-8") as fh:
        txt = fh.read()
    assert "-m src.serving.nightly" in txt and "--fallback-dir" in txt and "data/yedek" in txt
    assert "AI_CORE_PYTHON" in txt and ".venv" in txt
    assert "ysfygc" not in txt and "anaconda" not in txt


# ----------------------------------------------------------- Kol-B wiring
class _Stub:
    def predict(self, X):
        return np.asarray(X)[:, 0]


def _latest(n=20):
    return pd.DataFrame({"symbol": [f"S{i}" for i in range(n)],
                         "Date": pd.Timestamp("2026-06-03"),
                         "f0": np.arange(n, dtype=float), "Close": 100.0})


def _seg(n=20):
    return pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "liq_bucket": "Q1",
                         "vol_bucket": "Q5", "sector": "Ind"})


_MAPS = {"liq": {"Q1": 1.4}, "vol": {"Q5": 1.4}, "sector": {"Ind": 1.4}}


def test_vol_ratio_and_probability_source_passed_to_trend(monkeypatch):
    calls = []
    real = ns.trend_from_peer

    def spy(pct, n, cfg=None, symbol_vol_ratio=None):
        calls.append(symbol_vol_ratio)
        return real(pct, n, cfg, symbol_vol_ratio=symbol_vol_ratio)

    monkeypatch.setattr(ns, "trend_from_peer", spy)
    ratios = {f"S{i}": 0.5 + i / 10 for i in range(20)}
    out = ns.assemble_peer_table(_Stub(), _latest(), ["f0"], _seg(), icir_maps=_MAPS,
                                 symbol_vol_ratio=ratios)
    assert sorted(calls) == sorted(ratios.values())
    assert (out["probability_source"] == "oos_quintile_frequency").all()
    # yuksek vol orani -> daha genis bant
    lo = out.set_index("symbol")
    assert (lo.loc["S19", "kolb_price_high"] - lo.loc["S19", "kolb_price_low"]) > \
           (lo.loc["S18", "kolb_price_high"] - lo.loc["S18", "kolb_price_low"])


def test_n_eff_passed_to_peer_confidence_and_gates_high(monkeypatch):
    seen = []
    real = ns.peer_confidence

    def spy(*a, **k):
        seen.append(k.get("n_eff"))
        return real(*a, **k)

    monkeypatch.setattr(ns, "peer_confidence", spy)
    neff = {"liq": {"Q1": 120}, "vol": {"Q5": 30}, "sector": {"Ind": 200}}
    out = ns.assemble_peer_table(_Stub(), _latest(), ["f0"], _seg(), icir_maps=_MAPS, neff_maps=neff)
    assert set(seen) == {30}  # min over axes (muhafazakar)
    assert (out["confidence_label"] == "medium").all()  # ICIR 1.4 ama n_eff<50 -> high yok
    neff_ok = {k: {b: 100 for b in v} for k, v in neff.items()}
    out2 = ns.assemble_peer_table(_Stub(), _latest(), ["f0"], _seg(), icir_maps=_MAPS, neff_maps=neff_ok)
    assert (out2["confidence_label"] == "high").all()


def test_segment_icir_prefers_nonoverlap_and_neff_table():
    t = pd.DataFrame({"segment": ["Q1", "Q2", "Q3"],
                      "icir": [2.0, 1.0, 0.8],
                      "icir_nonoverlap": [0.9, np.nan, 0.5],
                      "n_nonoverlap": [40, 0, 70]})
    icir = ns.segment_icir_from_table(t)
    assert icir["Q1"] == pytest.approx(0.9) and icir["Q3"] == pytest.approx(0.5)
    assert icir["Q2"] == pytest.approx(1.0)  # nonoverlap NaN -> overlapping geri-dusus
    neff = ns.segment_neff_from_table(t)
    assert neff == {"Q1": 40, "Q2": None, "Q3": 70}
    # eski tablo (kolon yok) -> geriye uyumlu
    old = ns.segment_icir_from_table(t[["segment", "icir"]])
    assert old["Q1"] == pytest.approx(2.0)
    assert ns.segment_neff_from_table(t[["segment", "icir"]]) == {"Q1": None, "Q2": None, "Q3": None}


def test_symbol_vol_ratios_causal():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2026-01-01", periods=60)
    rows = []
    for i, sd in enumerate([0.01, 0.02, 0.04]):
        px = 100 * np.exp(np.cumsum(rng.normal(0, sd, len(dates))))
        rows += [{"symbol": f"S{i}", "Date": d, "Close": p} for d, p in zip(dates, px)]
    panel = pd.DataFrame(rows)
    as_of = dates[-1]
    r1 = ns.symbol_vol_ratios(panel, as_of)
    assert r1["S2"] > r1["S1"] > r1["S0"] and r1["S1"] == pytest.approx(1.0, abs=0.6)
    # gelecek satirlar eklense de as_of sonucu degismez (nedensel)
    fut = panel.copy()
    fut = pd.concat([fut, pd.DataFrame({"symbol": ["S0"], "Date": [as_of + pd.Timedelta(days=3)],
                                        "Close": [1e6]})])
    assert ns.symbol_vol_ratios(fut, as_of) == r1


def test_assemble_serving_table_passes_ratio_and_neff(monkeypatch):
    got = {}
    monkeypatch.setattr(nl, "assemble_peer_table",
                        lambda *a, **k: got.update(k) or pd.DataFrame())
    full = _latest().assign(Date=pd.Timestamp("2026-06-03"))
    monkeypatch.setattr(nl, "symbol_vol_ratios", lambda p, a: {"S1": 1.7})
    nl.assemble_serving_table(_Stub(), full, full, pd.Timestamp("2026-06-03"), ["f0"], _seg(),
                              _MAPS, {"liq": {"Q1": 77}})
    assert got["symbol_vol_ratio"] == {"S1": 1.7}
    assert got["neff_maps"] == {"liq": {"Q1": 77}} and got["icir_maps"] == _MAPS
