# -*- coding: utf-8 -*-
"""Gecelik kosunun makine-okunur durum dosyasi (data/nightly_status.json).

Amac: yarim kalan / olen bir kosu SESSIZ kalmasin. Dosya kosu BASINDA ``completed=false``
ile yazilir, her asamada guncellenir ve yalniz kosu normal biterse ``completed=true``
olur. Surec oldurulur/donarsa dosya ``completed=false`` kalir -> `/health` bunu
``interrupted`` olarak gosterir. Yazim atomiktir (gecici dosya + os.replace); yazim
hatasi kosuyu DURDURMAZ (uyari loglanir).

Yalniz stdlib: `/health` (API) bu modulu agir bagimlilik yuklemeden okuyabilir.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

STATUS_FILENAME = "nightly_status.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def status_path_for_db(db_path: str) -> str:
    """Durum dosyasi serving DB'nin yaninda durur (`data/serving_pool.db` -> `data/nightly_status.json`)."""
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), STATUS_FILENAME)


def write_status_atomic(path: str, payload: dict[str, Any]) -> None:
    """JSON'u ayni klasorde gecici dosyaya yazip os.replace ile yerine koyar (yari yazilmis dosya olmaz)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError as rm_exc:
                logger.warning("gecici durum dosyasi silinemedi (%s): %s", tmp, rm_exc)
        raise


def read_status(path: str) -> Optional[dict[str, Any]]:
    """Durum dosyasini okur; yok -> None; bozuk -> ValueError (cagiran karar verir)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"durum dosyasi okunamadi: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("durum dosyasi bir JSON nesnesi degil")
    return data


class NightlyRunStatus:
    """Kosu boyunca guncellenen durum belgesi. Olusturulur olusturulmaz diske yazilir (completed=false)."""

    def __init__(self, path: str, **initial: Any) -> None:
        self.path = path
        self.data: dict[str, Any] = {
            "started_at": _now_iso(),
            "finished_at": None,
            "completed": False,
            "phase": "started",
            "exit_code": None,
            "pid": os.getpid(),
            **initial,
        }
        self._flush()

    def update(self, **fields: Any) -> None:
        self.data.update(fields)
        self._flush()

    def finish(self, exit_code: int, **fields: Any) -> None:
        """Kosu normal sonlandi (basarili ya da bilincli atlama/ret): completed=true."""
        self.update(completed=True, finished_at=_now_iso(), exit_code=int(exit_code),
                    phase="finished", **fields)

    def fail(self, error: str, exit_code: int = 1) -> None:
        """Beklenmeyen istisna: bitis zamani yazilir ama completed=false kalir."""
        self.update(completed=False, finished_at=_now_iso(), exit_code=int(exit_code),
                    phase="crashed", error=error[:300])

    def _flush(self) -> None:
        self.data["updated_at"] = _now_iso()
        try:
            write_status_atomic(self.path, self.data)
        except OSError as exc:  # durum yazimi gecelik isi dusurmemeli
            logger.warning("durum dosyasi yazilamadi (%s): %s", self.path, exc)
