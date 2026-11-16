from src.serving.universe_seed import seed_missing_from_fallback


def test_seeds_only_active_symbols_missing_from_primary(tmp_path):
    data, fb = tmp_path / "data", tmp_path / "yedek"
    data.mkdir(); fb.mkdir()
    (data / "AKBNK.csv").write_text("primary")
    for sym in ("AKBNK", "KCHOL", "OLDCO"):
        (fb / f"{sym}.csv").write_text(f"fb-{sym}")
    uni = tmp_path / "u.csv"
    uni.write_text("Symbol,Status\nAKBNK,Active\nKCHOL,Active\nOLDCO,Delisted\nTUPRS,Active\n")

    seeded = seed_missing_from_fallback(str(data), str(uni), [str(fb)])

    assert seeded == ["KCHOL"]
    assert (data / "KCHOL.csv").read_text() == "fb-KCHOL" and (fb / "KCHOL.csv").exists()
    assert (data / "AKBNK.csv").read_text() == "primary"
    assert not (data / "OLDCO.csv").exists() and not (data / "TUPRS.csv").exists()


def test_unreadable_universe_is_noop(tmp_path):
    assert seed_missing_from_fallback(str(tmp_path), str(tmp_path / "missing.csv"), [str(tmp_path)]) == []
