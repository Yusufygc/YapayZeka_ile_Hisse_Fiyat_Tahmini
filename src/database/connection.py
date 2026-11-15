# -*- coding: utf-8 -*-
"""SQLite bağlantı yardımcısı.

`sqlite3.Connection` context manager'ı (`with conn:`) yalnızca commit/rollback
yapar, bağlantıyı KAPATMAZ. Projede yaygın olan `with self._connect() as conn:`
kalıbı bu yüzden dosya tanıtıcısını açık bırakıyordu; Python 3.10'da referans
sayımı bunu çoğu zaman gizliyordu, Python 3.13'te Windows'ta DB dosyası kilitli
kalıp taşıma/silme işlemleri `PermissionError` veriyordu.

`ClosingConnection`, `with` bloğundan çıkarken önce commit/rollback yapar, sonra
bağlantıyı kapatır. Bağlantıyı `with` olmadan kullanan çağıranlar yine
`close()` çağırmalıdır.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

BUSY_TIMEOUT_MS = 5000

_WAL_LOCK = threading.Lock()


class ClosingConnection(sqlite3.Connection):
    """`with` çıkışında commit/rollback sonrası kapanan SQLite bağlantısı."""

    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect(path: str, **kwargs) -> sqlite3.Connection:
    """`sqlite3.connect` ile aynı, ancak `ClosingConnection` döndürür."""
    kwargs.setdefault("factory", ClosingConnection)
    return sqlite3.connect(path, **kwargs)


def open_db(path: str, *, foreign_keys: bool = True, row_factory=sqlite3.Row) -> sqlite3.Connection:
    """Ortak bağlantı fabrikası: `busy_timeout` + WAL (yalnızca gerekirse yazılır).

    - `busy_timeout` bağlantı bazlıdır; her bağlantıda ayarlanır (yazar çakışmasında
      anında `database is locked` yerine kısa süre bekler).
    - `journal_mode=WAL` DB dosyasında kalıcıdır: mevcut mod okunur (kilitsiz) ve
      yalnızca WAL değilse bir kez ayarlanır; böylece her bağlantı yazma kilidi
      istemez, silinip yeniden oluşturulan dosyalar da doğru kurulur.
    """
    conn = sqlite3.connect(path, factory=ClosingConnection, timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = row_factory
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    if str(mode).lower() != "wal":
        with _WAL_LOCK:
            conn.execute("PRAGMA journal_mode=WAL")
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys=ON")
    return conn


def readonly_uri(path: str) -> str:
    """`file:` URI (`?mode=ro`); bosluk/#/? gibi karakterler yuzde-kodlanir."""
    return Path(os.path.abspath(path)).as_uri() + "?mode=ro"


def open_readonly(path: str, *, row_factory=sqlite3.Row) -> sqlite3.Connection:
    """Gercek salt-okunur baglanti: dosya OLUSTURMAZ, sema/journal_mode DEGISTIRMEZ.

    Dosya yoksa `sqlite3.OperationalError` (unable to open database file). `PRAGMA
    journal_mode=WAL` / `foreign_keys` / CREATE / ALTER calistirilmaz; yalniz
    `busy_timeout` ayarlanir. Yazma girisimi `attempt to write a readonly database` verir.
    """
    conn = sqlite3.connect(readonly_uri(path), uri=True, factory=ClosingConnection,
                           timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = row_factory
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn
