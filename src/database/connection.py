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

import sqlite3


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
