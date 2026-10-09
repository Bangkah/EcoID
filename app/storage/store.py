"""Local persistence: SQLite for records, plain files for photos. No network, stdlib only.

<data_dir>/ecoid.db
<data_dir>/images/<yyyy>/<mm>/<id>.<ext>   original photo, untouched (full resolution for human verification)
<data_dir>/thumbs/<id>.jpg                 small preview for the history list
<data_dir>/drafts/                         photos awaiting verification (managed by the observation manager)

Schema versions (PRAGMA user_version):  1 = Phase 3 record;  2 = + user_species, captured_at, location_source/accuracy.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Optional

from app.ai.contract.labels import COMMON_NAMES
from app.ai.contract.result import Status
from app.observation.models import Observation
from app.observation.verification import VerificationStatus

SCHEMA_VERSION = 2
UNSET = object()          # "leave this field alone" (None means "clear it")

_SCHEMA_V1 = """
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

_V2_COLUMNS = (
    "ALTER TABLE observations ADD COLUMN user_species TEXT",
    "ALTER TABLE observations ADD COLUMN captured_at TEXT",
    "ALTER TABLE observations ADD COLUMN location_source TEXT CHECK (location_source IS NULL OR location_source IN ('exif','device','manual'))",
    "ALTER TABLE observations ADD COLUMN location_accuracy_m REAL",
)

_COLUMNS = ("id image_path predicted_species confidence alternative_predictions verification_status "
            "timestamp latitude longitude notes model identification_status "
            "user_species captured_at location_source location_accuracy_m").split()


def _like(s: str) -> str:
    return "%" + s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


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
            if v > SCHEMA_VERSION:
                raise RuntimeError(f"Database schema v{v} is newer than this app (v{SCHEMA_VERSION}).")
            if v == 0:
                c.executescript(_SCHEMA_V1)
                v = 1
            if v == 1:
                for stmt in _V2_COLUMNS:
                    c.execute(stmt)
                # Phase 3 allowed coordinates without recording where they came from; v2 requires a source.
                c.execute("UPDATE observations SET location_source = 'manual' WHERE latitude IS NOT NULL")
            c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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
            user_species=r["user_species"], captured_at=r["captured_at"],
            location_source=r["location_source"], location_accuracy_m=r["location_accuracy_m"],
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

    def list(self, *, status: Optional[VerificationStatus] = None, q: Optional[str] = None,
             species: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None,
             has_location: Optional[bool] = None, limit: int = 50, offset: int = 0):
        """Newest first. Returns (items, total_matching).

        q: case-insensitive text in notes / species (scientific or common name) / the user's correction.
        species: exact scientific name, matched against the suggestion or the user's correction.
        date_from/date_to: 'YYYY-MM-DD', inclusive, on the capture date (falls back to the save date).
        """
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        where, args = [], []
        if status:
            where.append("verification_status = ?")
            args.append(status.value)
        if q and q.strip():
            term = q.strip()
            commons = [sci for sci, com in COMMON_NAMES.items() if term.lower() in com.lower()]
            clauses = ["notes LIKE ? ESCAPE '\\'", "predicted_species LIKE ? ESCAPE '\\'", "user_species LIKE ? ESCAPE '\\'"]
            args += [_like(term)] * 3
            if commons:
                clauses.append(f"predicted_species IN ({','.join('?' * len(commons))})")
                args += commons
            where.append("(" + " OR ".join(clauses) + ")")
        if species:
            where.append("(predicted_species = ? COLLATE NOCASE OR user_species = ? COLLATE NOCASE)")
            args += [species, species]
        day = "substr(COALESCE(captured_at, timestamp), 1, 10)"
        if date_from:
            where.append(f"{day} >= ?")
            args.append(date_from)
        if date_to:
            where.append(f"{day} <= ?")
            args.append(date_to)
        if has_location is True:
            where.append("latitude IS NOT NULL")
        elif has_location is False:
            where.append("latitude IS NULL")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM observations {clause}", args).fetchone()[0]
            rows = c.execute(f"SELECT * FROM observations {clause} ORDER BY timestamp DESC, id LIMIT ? OFFSET ?",
                             args + [limit, offset]).fetchall()
        return [self._row_to_obs(r) for r in rows], total

    def list_all(self, **filters) -> list:
        """Every observation matching the filters (pages of 200 internally). For stats / map / export."""
        out, offset = [], 0
        while True:
            items, total = self.list(limit=200, offset=offset, **filters)
            out += items
            offset += len(items)
            if not items or offset >= total:
                return out

    def update(self, obs_id: str, *, verification_status: Optional[VerificationStatus] = None,
               notes: Optional[str] = None, user_species=UNSET, location=UNSET) -> Observation:
        """location: None clears it; (lat, lon, source, accuracy_m) sets it; UNSET leaves it.
        Switching to VERIFIED automatically drops a previous correction."""
        cur = self.get(obs_id)  # raises NotFoundError
        new_status = verification_status or cur.verification_status
        changes = {"verification_status": new_status, "notes": cur.notes if notes is None else notes}
        if user_species is not UNSET:
            changes["user_species"] = user_species
        elif new_status == VerificationStatus.VERIFIED:
            changes["user_species"] = None
        if location is not UNSET:
            if location is None:
                changes.update(latitude=None, longitude=None, location_source=None, location_accuracy_m=None)
            else:
                lat, lon, src, acc = location
                changes.update(latitude=lat, longitude=lon, location_source=src, location_accuracy_m=acc)
        new = replace(cur, **changes)          # validates everything before the DB is touched
        with self._conn() as c:
            c.execute("UPDATE observations SET verification_status=?, notes=?, user_species=?, latitude=?, longitude=?, "
                      "location_source=?, location_accuracy_m=? WHERE id=?",
                      (new.verification_status.value, new.notes, new.user_species, new.latitude, new.longitude,
                       new.location_source, new.location_accuracy_m, obs_id))
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
