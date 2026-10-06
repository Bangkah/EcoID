"""Local persistence: SQLite for records, plain files for photos. No network, stdlib only.

<data_dir>/ecoid.db
<data_dir>/images/<yyyy>/<mm>/<id>.<ext>   original photo, untouched (full resolution for human verification)
<data_dir>/thumbs/<id>.jpg                 small preview for the history list
<data_dir>/drafts/                         photos awaiting verification (managed by the observation manager)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from app.ai.contract.result import Status
from app.observation.models import Observation
from app.observation.verification import VerificationStatus

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE observations (
    id                      TEXT PRIMARY KEY,
    image_path              TEXT NOT NULL,
    predicted_species       TEXT NOT NULL,
    confidence              REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    alternative_predictions TEXT NOT NULL,
    verification_status     TEXT NOT NULL CHECK (verification_status IN ('VERIFIED','REJECTED','UNCERTAIN')),
    timestamp               TEXT NOT NULL,
    latitude                REAL CHECK (latitude  IS NULL OR latitude  BETWEEN -90  AND 90),
    longitude               REAL CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180),
    notes                   TEXT NOT NULL DEFAULT '',
    model                   TEXT NOT NULL,
    identification_status   TEXT NOT NULL CHECK (identification_status IN ('IDENTIFIED','LOW_CONFIDENCE'))
);
CREATE INDEX idx_observations_timestamp ON observations (timestamp DESC, id);
"""

_COLUMNS = ("id image_path predicted_species confidence alternative_predictions verification_status "
            "timestamp latitude longitude notes model identification_status").split()


class NotFoundError(KeyError):
    pass


class ObservationStore:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        for sub in ("images", "thumbs", "drafts"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "ecoid.db"
        self._migrate()

    # -- connection per operation: simple and safe with a threaded server ----------------
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate(self):
        with self._conn() as c:
            v = c.execute("PRAGMA user_version").fetchone()[0]
            if v == 0:
                c.executescript(_SCHEMA)
                c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif v > SCHEMA_VERSION:
                raise RuntimeError(f"Database schema v{v} is newer than this app (v{SCHEMA_VERSION}).")

    # -- paths ----------------------------------------------------------------------------
    def abspath(self, rel: str) -> Path:
        p = (self.data_dir / rel).resolve()
        if self.data_dir.resolve() not in p.parents:
            raise ValueError("path escapes data dir")
        return p

    def thumb_path(self, obs_id: str) -> Path:
        return self.data_dir / "thumbs" / f"{obs_id}.jpg"

    # -- CRUD -----------------------------------------------------------------------------
    @staticmethod
    def _row_to_obs(r: sqlite3.Row) -> Observation:
        return Observation(
            id=r["id"], image_path=r["image_path"], predicted_species=r["predicted_species"],
            confidence=r["confidence"],
            alternative_predictions=tuple(json.loads(r["alternative_predictions"])),
            verification_status=VerificationStatus(r["verification_status"]),
            timestamp=r["timestamp"], latitude=r["latitude"], longitude=r["longitude"],
            notes=r["notes"], model=r["model"], identification_status=Status(r["identification_status"]),
        )

    def insert(self, obs: Observation) -> None:
        d = obs.to_dict()
        d["alternative_predictions"] = json.dumps(d["alternative_predictions"])
        with self._conn() as c:
            c.execute(f"INSERT INTO observations ({','.join(_COLUMNS)}) VALUES ({','.join('?' * len(_COLUMNS))})",
                      [d[k] for k in _COLUMNS])

    def get(self, obs_id: str) -> Observation:
        with self._conn() as c:
            r = c.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        if r is None:
            raise NotFoundError(obs_id)
        return self._row_to_obs(r)

    def list(self, *, status: Optional[VerificationStatus] = None, limit: int = 50, offset: int = 0):
        """Newest first. Returns (items, total_matching)."""
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        where, args = ("WHERE verification_status = ?", [status.value]) if status else ("", [])
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM observations {where}", args).fetchone()[0]
            rows = c.execute(f"SELECT * FROM observations {where} ORDER BY timestamp DESC, id LIMIT ? OFFSET ?",
                             args + [limit, offset]).fetchall()
        return [self._row_to_obs(r) for r in rows], total

    def update(self, obs_id: str, *, verification_status: Optional[VerificationStatus] = None,
               notes: Optional[str] = None) -> Observation:
        cur = self.get(obs_id)  # raises NotFoundError
        new_status = verification_status or cur.verification_status
        new_notes = cur.notes if notes is None else notes
        # re-validate through the dataclass before touching the DB
        from dataclasses import replace
        replace(cur, verification_status=new_status, notes=new_notes)
        with self._conn() as c:
            c.execute("UPDATE observations SET verification_status = ?, notes = ? WHERE id = ?",
                      (new_status.value, new_notes, obs_id))
        return self.get(obs_id)

    def delete(self, obs_id: str) -> None:
        obs = self.get(obs_id)
        with self._conn() as c:
            c.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
        for p in (self.abspath(obs.image_path), self.thumb_path(obs_id)):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
