import sqlite3

import pytest

from src.database.connection import ClosingConnection, connect


def test_closing_connection_closes_on_with_exit(tmp_path):
    db = tmp_path / "x.db"
    conn = connect(str(db))
    with conn:
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    # Commit gerçekleşmiş olmalı ve dosya taşınabilir olmalı.
    moved = tmp_path / "y.db"
    db.rename(moved)
    with sqlite3.connect(str(moved), factory=ClosingConnection) as check:
        assert check.execute("SELECT a FROM t").fetchone()[0] == 1


def test_closing_connection_rolls_back_on_error(tmp_path):
    db = tmp_path / "x.db"
    with connect(str(db)) as conn:
        conn.execute("CREATE TABLE t (a INTEGER)")
    with pytest.raises(RuntimeError):
        with connect(str(db)) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise RuntimeError("boom")
    with connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
