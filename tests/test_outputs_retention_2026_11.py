# -*- coding: utf-8 -*-
"""outputs/ retention, hafif latest/, baslangic manifesti ve sikistirilmis model testleri."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import types

import joblib
import numpy as np
import pytest

from src.pipeline.artifacts import write_run_manifest
from src.utils import output_retention as ret
from src.utils.output_paths import (
    LATEST_POINTER_NAME,
    read_latest_pointer,
    resolve_latest_artifact,
    run_id_from_path,
    sync_latest_output,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_prune_tool():
    spec = importlib.util.spec_from_file_location(
        "prune_outputs_tool", os.path.join(_ROOT, "tools", "prune_outputs.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(symbol_root, run_id, *, model_bytes=2048, status=None):
    run_dir = os.path.join(symbol_root, "runs", run_id)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "csv"), exist_ok=True)
    with open(os.path.join(run_dir, "models", "m.pkl"), "wb") as fh:
        fh.write(b"x" * model_bytes)
    with open(os.path.join(run_dir, "csv", "r.csv"), "w", encoding="utf-8") as fh:
        fh.write("a;b\n1;2\n")
    manifest = {"run_id": run_id}
    if status:
        manifest["status"] = status
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return run_dir


# ---------------------------------------------------------------- compress


def test_compressed_dump_loads_back_and_is_smaller(tmp_path):
    obj = {"w": np.zeros(200_000), "name": "rf"}
    plain = tmp_path / "plain.pkl"
    packed = tmp_path / "packed.pkl"
    joblib.dump(obj, plain)
    joblib.dump(obj, packed, compress=3)
    assert os.path.getsize(packed) < os.path.getsize(plain) / 10
    loaded = joblib.load(packed)
    assert loaded["name"] == "rf" and np.array_equal(loaded["w"], obj["w"])


def test_model_save_methods_use_compression(tmp_path):
    import inspect
    import src.models.xgboost_model as xgb_mod
    import src.models.linear_model as lin_mod

    for mod in (xgb_mod, lin_mod):
        assert "compress=3" in inspect.getsource(mod)


def test_compress_file_atomic_and_verified(tmp_path):
    path = tmp_path / "big.pkl"
    obj = {"w": np.arange(300_000, dtype=float)}
    joblib.dump(obj, path)
    before = os.path.getsize(path)
    saved = ret.compress_file(str(path))
    assert saved > 0 and os.path.getsize(path) == before - saved
    assert not os.path.exists(str(path) + ".compress.tmp")
    assert np.array_equal(joblib.load(path)["w"], obj["w"])


# ---------------------------------------------------------------- latest


def test_latest_pointer_and_resolve_model_in_run_dir(tmp_path):
    symbol_root = str(tmp_path / "AAA")
    run_id = "20260101_000000_AAA_walk_forward_models-All"
    run_dir = _make_run(symbol_root, run_id, model_bytes=6 * 1024 * 1024)
    big_csv = os.path.join(run_dir, "csv", "huge.csv")
    with open(big_csv, "wb") as fh:
        fh.write(b"0" * (6 * 1024 * 1024))
    latest = os.path.join(symbol_root, "latest")
    os.makedirs(latest)
    with open(os.path.join(latest, "old.txt"), "w", encoding="utf-8") as fh:
        fh.write("old")

    sync_latest_output(symbol_root, run_dir, latest, run_id)

    pointer = read_latest_pointer(symbol_root)
    assert pointer["run_id"] == run_id
    assert pointer["path"] == f"runs/{run_id}"
    assert "created_at" in pointer
    assert os.path.isfile(os.path.join(latest, LATEST_POINTER_NAME))
    assert os.path.isfile(os.path.join(latest, "csv", "r.csv"))  # kucuk dosya kopyalandi
    assert os.path.isfile(os.path.join(latest, "run_manifest.json"))
    assert not os.path.exists(os.path.join(latest, "models", "m.pkl"))  # model kopyalanmadi
    assert not os.path.exists(os.path.join(latest, "csv", "huge.csv"))  # esik ustu
    assert not os.path.exists(os.path.join(latest, "old.txt"))
    assert not os.path.exists(os.path.join(symbol_root, ".latest_sync.lock"))

    resolved = resolve_latest_artifact(symbol_root, "models/m.pkl")
    assert resolved is not None
    assert os.path.normpath(resolved) == os.path.normpath(os.path.join(run_dir, "models", "m.pkl"))
    assert resolve_latest_artifact(symbol_root, "csv/r.csv").endswith("r.csv")
    assert resolve_latest_artifact(symbol_root, "../../etc/passwd") is None
    assert resolve_latest_artifact(symbol_root, "models/missing.pkl") is None


def test_run_id_from_path_handles_windows_and_posix():
    rid = "20260523_135857_AKBNK_walk_forward_model-X"
    assert run_id_from_path(f"D:\\x\\outputs\\AKBNK\\runs\\{rid}\\models\\a.pkl") == rid
    assert run_id_from_path(f"/x/outputs/AKBNK/runs/{rid}/models/a.pkl") == rid
    assert run_id_from_path("/x/outputs/AKBNK/latest/models/a.pkl") is None


# ---------------------------------------------------------------- manifest


def _fake_pipeline(tmp_path):
    out = tmp_path / "out" / "TST" / "runs" / "20260101_000000_TST_wf"
    out.mkdir(parents=True)
    data_file = tmp_path / "TST.csv"
    data_file.write_text("Date,Close\n2026-01-01,1\n", encoding="utf-8")
    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(feature_mode="v2"),
        execution=types.SimpleNamespace(signal_config=object()),
    )
    return types.SimpleNamespace(
        run_id="20260101_000000_TST_wf",
        outputs_dir=str(out),
        project_root=str(tmp_path),
        stock_symbol="TST",
        data_file=str(data_file),
        _cfg=cfg,
        model_config={},
        quality_gate_mode="soft",
        min_directional_accuracy=52.0,
        max_rmse_vs_benchmark=1.0,
        min_composite_score=0.0,
        candidate_models={"XGBoost"},
        validation_mode="walk_forward",
    )


def _read_manifest(pipeline):
    with open(os.path.join(pipeline.outputs_dir, "run_manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_manifest_running_then_completed_keeps_started_at(tmp_path):
    p = _fake_pipeline(tmp_path)
    write_run_manifest(p, status="running")
    first = _read_manifest(p)
    assert first["status"] == "running" and "finished_at" not in first
    write_run_manifest(p, status="completed")
    second = _read_manifest(p)
    assert second["status"] == "completed"
    assert second["started_at"] == first["started_at"]
    assert "finished_at" in second


def _orchestrator_stub(tmp_path, body):
    from src.pipeline.orchestrator import ForecastingPipeline

    p = _fake_pipeline(tmp_path)
    p.output_root = str(tmp_path / "out" / "TST")
    p.latest_dir = os.path.join(p.output_root, "latest")
    p.setup_environment = lambda: None
    p._execute_run = body
    for name in ("run_all", "_write_run_manifest", "_write_running_manifest",
                 "_finalize_manifest", "_sync_latest_output"):
        setattr(p, name, types.MethodType(getattr(ForecastingPipeline, name), p))
    return p


def test_run_all_completes_manifest_and_syncs_latest(tmp_path):
    p = _orchestrator_stub(tmp_path, lambda: None)
    seen = {}

    def body():
        seen["during"] = _read_manifest(p)["status"]

    p._execute_run = body
    p.run_all()
    assert seen["during"] == "running"
    assert _read_manifest(p)["status"] == "completed"
    assert read_latest_pointer(p.output_root)["run_id"] == p.run_id


def test_run_all_exception_leaves_non_completed_manifest(tmp_path):
    def boom():
        raise RuntimeError("cokme")

    p = _orchestrator_stub(tmp_path, boom)
    with pytest.raises(RuntimeError):
        p.run_all()
    manifest = _read_manifest(p)
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert read_latest_pointer(p.output_root) is None  # basarisiz run latest'e yazilmaz


def test_hard_crash_leaves_running(tmp_path):
    p = _fake_pipeline(tmp_path)
    write_run_manifest(p, status="running")
    # surec oldu: ikinci yazim yok
    assert _read_manifest(p)["status"] == "running"


# ---------------------------------------------------------------- prune


def _build_outputs(tmp_path):
    outputs = tmp_path / "outputs"
    root = str(outputs / "AAA")
    ids = [f"2026010{i}_000000_AAA_walk_forward_models-All" for i in range(1, 9)]  # 8 run
    for rid in ids:
        _make_run(root, rid)
    # DB sadece en eski run'i referans verir (Windows tarzi yol), latest orta bir run'i
    db = tmp_path / "stock_models.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE best_models (id INTEGER PRIMARY KEY, model_path TEXT)")
    conn.execute("CREATE TABLE experiments (id INTEGER PRIMARY KEY, model_path TEXT)")
    conn.execute(
        "INSERT INTO best_models(model_path) VALUES (?)",
        (f"D:\\other\\outputs\\AAA\\runs\\{ids[0]}\\models\\m.pkl",),
    )
    conn.commit()
    conn.close()
    latest = os.path.join(root, "latest")
    os.makedirs(latest)
    os.makedirs(os.path.join(root, "latest"), exist_ok=True)
    from src.utils.output_paths import write_latest_pointer

    write_latest_pointer(root, ids[1])
    return str(outputs), str(db), ids


def test_prune_keeps_n_db_and_latest_dry_run_deletes_nothing(tmp_path):
    outputs, db, ids = _build_outputs(tmp_path)
    report = ret.plan_prune(outputs, keep=3, db_path=db)
    kept = {r.run_id for r in report.runs if r.keep}
    # son 3 (ids[5..7]) + db (ids[0]) + latest (ids[1])
    assert kept == {ids[0], ids[1], ids[5], ids[6], ids[7]}
    assert {r.run_id for r in report.to_delete} == {ids[2], ids[3], ids[4]}
    assert report.bytes_to_free > 0
    assert all(os.path.isdir(os.path.join(outputs, "AAA", "runs", i)) for i in ids)


def test_prune_cli_dry_run_default_and_apply(tmp_path, capsys):
    tool = _load_prune_tool()
    outputs, db, ids = _build_outputs(tmp_path)
    argv = ["--outputs-dir", outputs, "--db", db, "--keep", "3"]

    assert tool.main(argv) == 0
    assert "DRY-RUN" in capsys.readouterr().out
    assert all(os.path.isdir(os.path.join(outputs, "AAA", "runs", i)) for i in ids)

    assert tool.main(argv + ["--apply"]) == 0
    remaining = set(os.listdir(os.path.join(outputs, "AAA", "runs")))
    assert remaining == {ids[0], ids[1], ids[5], ids[6], ids[7]}


def test_prune_protects_recent_running_run(tmp_path):
    outputs = tmp_path / "outputs"
    root = str(outputs / "BBB")
    ids = [f"2026010{i}_000000_BBB_walk_forward_models-All" for i in range(1, 4)]
    for rid in ids:
        _make_run(root, rid, status="running" if rid == ids[0] else "completed")
    report = ret.plan_prune(str(outputs), keep=1, db_path=None)
    assert {r.run_id for r in report.to_delete} == {ids[1]}


def test_prune_cli_rejects_apply_with_dry_run(tmp_path):
    tool = _load_prune_tool()
    assert tool.main(["--outputs-dir", str(tmp_path), "--apply", "--dry-run"]) == 2


def test_slim_latest_legacy_full_copy(tmp_path):
    root = str(tmp_path / "outputs" / "CCC")
    rid = "20260101_000000_CCC_walk_forward_models-All"
    run_dir = _make_run(root, rid)
    import shutil

    shutil.copytree(run_dir, os.path.join(root, "latest"))
    dry = ret.slim_latest(str(tmp_path / "outputs"), apply=False)
    assert dry["symbols"] == 1 and dry["bytes"] > 0
    assert os.path.exists(os.path.join(root, "latest", "models", "m.pkl"))
    ret.slim_latest(str(tmp_path / "outputs"), apply=True)
    assert not os.path.exists(os.path.join(root, "latest", "models", "m.pkl"))
    assert read_latest_pointer(root)["run_id"] == rid
    assert resolve_latest_artifact(root, "models/m.pkl") is not None


def test_compress_file_accepts_tree_ensemble_with_unstable_hash(tmp_path, monkeypatch):
    """RandomForest joblib.hash'i yeniden yüklemede değişebilir; ağaç eşitliği yeterli."""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from src.utils import output_retention as orr

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), rng.normal(size=60)
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)
    path = tmp_path / "rf.pkl"
    joblib.dump(model, path)
    counter = iter(range(10**6))
    monkeypatch.setattr(joblib, "hash", lambda obj: next(counter))  # her çağrı farklı
    orr.compress_file(str(path))
    reloaded = joblib.load(path)
    assert np.array_equal(reloaded.predict(X), model.predict(X))
    assert open(path, "rb").read(1) != b"\x80"
