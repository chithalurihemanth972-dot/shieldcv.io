"""
SQLite persistence layer for SHIELD-CV (``shield.db``).

Stores scans, findings, provenance records, contributor risk and an append-only
audit log. Every statement uses parameterised queries — no string interpolation
of user/vendor data anywhere.

The audit log is itself hash-chained: each row carries the SHA-256 of the
previous row, so silent deletion or edit of an audit entry is detectable via
:meth:`ShieldDatabase.verify_audit_chain`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.config import get_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
    scan_id       TEXT PRIMARY KEY,
    scan_type     TEXT NOT NULL,
    target        TEXT NOT NULL,
    started_ns    INTEGER NOT NULL,
    finished_ns   INTEGER,
    status        TEXT NOT NULL DEFAULT 'RUNNING',
    num_findings  INTEGER NOT NULL DEFAULT 0,
    risk_score    REAL NOT NULL DEFAULT 0.0,
    verdict       TEXT,
    config_json   TEXT,
    summary_json  TEXT
);

CREATE TABLE IF NOT EXISTS findings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id     TEXT NOT NULL,
    scan_id        TEXT NOT NULL,
    attack_class   TEXT NOT NULL,
    affected_asset TEXT NOT NULL,
    severity       TEXT NOT NULL,
    confidence     REAL NOT NULL,
    reason         TEXT NOT NULL,
    evidence_json  TEXT NOT NULL,
    disposition    TEXT NOT NULL,
    contributor    TEXT,
    module         TEXT,
    created_ns     INTEGER NOT NULL,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_findings_scan ON findings(scan_id);
CREATE INDEX IF NOT EXISTS idx_findings_class ON findings(attack_class);
CREATE INDEX IF NOT EXISTS idx_findings_contrib ON findings(contributor);

CREATE TABLE IF NOT EXISTS contributors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id       TEXT NOT NULL,
    name          TEXT NOT NULL,
    risk_score    REAL NOT NULL,
    num_findings  INTEGER NOT NULL,
    num_samples   INTEGER NOT NULL,
    attack_types  TEXT NOT NULL,
    created_ns    INTEGER NOT NULL,
    UNIQUE(scan_id, name)
);

CREATE TABLE IF NOT EXISTS provenance_records (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id        TEXT NOT NULL UNIQUE,
    sequence_number  INTEGER NOT NULL,
    input_hash       TEXT NOT NULL,
    model_hash       TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    output_hash      TEXT NOT NULL,
    timestamp_ns     INTEGER NOT NULL,
    nonce            TEXT NOT NULL,
    prev_record_hash TEXT NOT NULL,
    record_hash      TEXT NOT NULL,
    signature        TEXT NOT NULL,
    payload_json     TEXT,
    ingested_ns      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prov_seq ON provenance_records(sequence_number);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prov_nonce ON provenance_records(nonce);

CREATE TABLE IF NOT EXISTS model_registry (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_path   TEXT NOT NULL,
    weight_hash  TEXT NOT NULL,
    file_hash    TEXT,
    framework    TEXT,
    access_level TEXT,
    registered_ns INTEGER NOT NULL,
    notes        TEXT,
    UNIQUE(model_path, weight_hash)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ns INTEGER NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    target      TEXT,
    details     TEXT,
    prev_hash   TEXT NOT NULL,
    entry_hash  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp_ns);
"""

_GENESIS_HASH = "0" * 64

# Defence-in-depth: whitelist of valid table names.  The ``stats()`` and
# ``reset()`` methods interpolate table names into SQL, which is safe only when
# every candidate is hardcoded.  If the list is ever externalised (config, user
# input) this whitelist is the last guard against SQL injection.
_VALID_TABLES = frozenset({
    "scans", "findings", "contributors", "provenance_records",
    "model_registry", "audit_log", "schema_meta",
})


class ShieldDatabase:
    """Thread-safe SQLite wrapper for all SHIELD-CV persistence.

    Attributes:
        path: Absolute path of the database file.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        """Open (and if necessary create) the SHIELD-CV database.

        Args:
            path: Explicit database path. Defaults to ``paths.database`` in config.
        """
        cfg = get_config()
        self.path = Path(path) if path else cfg.path("database", "shield.db")
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
            LOGGER.debug("Database ready at %s", self.path)
        except Exception as exc:
            LOGGER.error("Database initialisation failed at %s: %s", self.path, exc)
            raise

    # -- connection management --------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection inside a transaction.

        Yields:
            An open :class:`sqlite3.Connection` with row factory set; commits on
            clean exit and rolls back on exception.
        """
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                yield conn
                conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create all tables and record the schema version."""
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("created_ns", str(time.time_ns())),
            )

    # -- scans -------------------------------------------------------------
    def create_scan(self, scan_id: str, scan_type: str, target: str,
                    config_snapshot: Optional[Dict[str, Any]] = None) -> bool:
        """Register the start of a scan.

        Args:
            scan_id: Unique scan identifier.
            scan_type: ``data`` | ``model`` | ``records`` | ``drift`` | ``office``.
            target: Path or descriptor of what was scanned.
            config_snapshot: Effective configuration for reproducibility.

        Returns:
            ``True`` on success.
        """
        try:
            with self.connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO scans "
                    "(scan_id, scan_type, target, started_ns, status, config_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (scan_id, scan_type, str(target), time.time_ns(), "RUNNING",
                     json.dumps(config_snapshot or {}, default=str)),
                )
            self.append_audit("SHIELD-CV", "SCAN_START", str(target),
                              {"scan_id": scan_id, "type": scan_type})
            return True
        except Exception as exc:
            LOGGER.error("create_scan failed: %s", exc)
            return False

    def complete_scan(self, scan_id: str, num_findings: int, risk_score: float,
                      verdict: str, summary: Optional[Dict[str, Any]] = None) -> bool:
        """Mark a scan finished and persist its headline results.

        Args:
            scan_id: Scan identifier.
            num_findings: Total findings produced.
            risk_score: Aggregate risk in ``[0, 1]``.
            verdict: Final verdict string.
            summary: Summary payload stored as JSON.

        Returns:
            ``True`` on success.
        """
        try:
            with self.connect() as conn:
                conn.execute(
                    "UPDATE scans SET finished_ns=?, status=?, num_findings=?, "
                    "risk_score=?, verdict=?, summary_json=? WHERE scan_id=?",
                    (time.time_ns(), "COMPLETE", int(num_findings), float(risk_score),
                     str(verdict), json.dumps(summary or {}, default=str), scan_id),
                )
            self.append_audit("SHIELD-CV", "SCAN_COMPLETE", scan_id,
                              {"findings": num_findings, "risk": risk_score,
                               "verdict": verdict})
            return True
        except Exception as exc:
            LOGGER.error("complete_scan failed: %s", exc)
            return False

    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one scan row.

        Args:
            scan_id: Scan identifier.

        Returns:
            Row as a dict, or ``None``.
        """
        try:
            with self.connect() as conn:
                row = conn.execute("SELECT * FROM scans WHERE scan_id=?", (scan_id,)).fetchone()
            return dict(row) if row else None
        except Exception as exc:
            LOGGER.error("get_scan failed: %s", exc)
            return None

    def list_scans(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent scans, newest first.

        Args:
            limit: Maximum rows to return.

        Returns:
            List of scan dictionaries.
        """
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM scans ORDER BY started_ns DESC LIMIT ?", (int(limit),)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            LOGGER.error("list_scans failed: %s", exc)
            return []

    # -- findings ----------------------------------------------------------
    def insert_findings(self, scan_id: str, findings: List[Dict[str, Any]],
                        module: str = "") -> int:
        """Bulk-insert findings for a scan.

        Args:
            scan_id: Owning scan identifier.
            findings: Finding dictionaries following the SHIELD-CV schema.
            module: Producing module name.

        Returns:
            Number of rows inserted.
        """
        if not findings:
            return 0
        try:
            now = time.time_ns()
            rows = []
            for finding in findings:
                rows.append((
                    str(finding.get("finding_id", "")),
                    scan_id,
                    str(finding.get("attack_class", "UNKNOWN")),
                    str(finding.get("affected_asset", "")),
                    str(finding.get("severity", "LOW")),
                    float(finding.get("confidence", 0.0)),
                    str(finding.get("reason", "")),
                    json.dumps(finding.get("evidence", {}), default=str),
                    str(finding.get("disposition", "ACCEPT")),
                    finding.get("contributor"),
                    module or finding.get("module", ""),
                    now,
                ))
            with self.connect() as conn:
                conn.executemany(
                    "INSERT INTO findings (finding_id, scan_id, attack_class, affected_asset, "
                    "severity, confidence, reason, evidence_json, disposition, contributor, "
                    "module, created_ns) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
            return len(rows)
        except Exception as exc:
            LOGGER.error("insert_findings failed: %s", exc)
            return 0

    def get_findings(self, scan_id: Optional[str] = None,
                     attack_class: Optional[str] = None,
                     severity: Optional[str] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """Query findings with optional filters.

        Args:
            scan_id: Restrict to one scan.
            attack_class: Restrict to one attack class.
            severity: Restrict to one severity level.
            limit: Maximum rows returned.

        Returns:
            Findings with ``evidence`` decoded back into a dict.
        """
        try:
            clauses: List[str] = []
            params: List[Any] = []
            if scan_id:
                clauses.append("scan_id = ?")
                params.append(scan_id)
            if attack_class:
                clauses.append("attack_class = ?")
                params.append(attack_class)
            if severity:
                clauses.append("severity = ?")
                params.append(severity)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(int(limit))
            with self.connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM findings{where} ORDER BY confidence DESC LIMIT ?",
                    tuple(params),
                ).fetchall()
            output: List[Dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                try:
                    item["evidence"] = json.loads(item.pop("evidence_json", "{}"))
                except Exception:
                    item["evidence"] = {}
                output.append(item)
            return output
        except Exception as exc:
            LOGGER.error("get_findings failed: %s", exc)
            return []

    # -- contributors ------------------------------------------------------
    def upsert_contributor_risk(self, scan_id: str, name: str, risk_score: float,
                                num_findings: int, num_samples: int,
                                attack_types: List[str]) -> bool:
        """Insert or update a contributor's aggregated risk for a scan.

        Args:
            scan_id: Owning scan.
            name: Contributor name.
            risk_score: Risk in ``[0, 1]``.
            num_findings: Findings attributed to the contributor.
            num_samples: Samples supplied by the contributor.
            attack_types: Distinct attack classes observed.

        Returns:
            ``True`` on success.
        """
        try:
            with self.connect() as conn:
                conn.execute(
                    "INSERT INTO contributors (scan_id, name, risk_score, num_findings, "
                    "num_samples, attack_types, created_ns) VALUES (?,?,?,?,?,?,?) "
                    "ON CONFLICT(scan_id, name) DO UPDATE SET risk_score=excluded.risk_score, "
                    "num_findings=excluded.num_findings, num_samples=excluded.num_samples, "
                    "attack_types=excluded.attack_types",
                    (scan_id, name, float(risk_score), int(num_findings), int(num_samples),
                     json.dumps(list(attack_types)), time.time_ns()),
                )
            return True
        except Exception as exc:
            LOGGER.error("upsert_contributor_risk failed: %s", exc)
            return False

    def get_contributor_risks(self, scan_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch contributor risk rows, highest risk first.

        Args:
            scan_id: Restrict to one scan.

        Returns:
            Contributor rows with ``attack_types`` decoded to a list.
        """
        try:
            with self.connect() as conn:
                if scan_id:
                    rows = conn.execute(
                        "SELECT * FROM contributors WHERE scan_id=? ORDER BY risk_score DESC",
                        (scan_id,)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM contributors ORDER BY risk_score DESC").fetchall()
            output = []
            for row in rows:
                item = dict(row)
                try:
                    item["attack_types"] = json.loads(item.get("attack_types") or "[]")
                except Exception:
                    item["attack_types"] = []
                output.append(item)
            return output
        except Exception as exc:
            LOGGER.error("get_contributor_risks failed: %s", exc)
            return []

    # -- provenance --------------------------------------------------------
    def insert_provenance_record(self, record: Dict[str, Any]) -> bool:
        """Persist one provenance record in the tamper-evident store.

        A duplicate nonce violates a UNIQUE index and is rejected — this is the
        database-level replay guard.

        Args:
            record: Record dictionary with canonical provenance fields.

        Returns:
            ``True`` if stored, ``False`` on duplicate or error.
        """
        try:
            with self.connect() as conn:
                conn.execute(
                    "INSERT INTO provenance_records (record_id, sequence_number, input_hash, "
                    "model_hash, config_hash, output_hash, timestamp_ns, nonce, "
                    "prev_record_hash, record_hash, signature, payload_json, ingested_ns) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(record.get("record_id", "")),
                        int(record.get("sequence_number", 0)),
                        str(record.get("input_hash", "")),
                        str(record.get("model_hash", "")),
                        str(record.get("config_hash", "")),
                        str(record.get("output_hash", "")),
                        int(record.get("timestamp_ns", 0)),
                        str(record.get("nonce", "")),
                        str(record.get("prev_record_hash", "")),
                        str(record.get("record_hash", "")),
                        str(record.get("signature", "")),
                        json.dumps(record.get("output"), default=str),
                        time.time_ns(),
                    ),
                )
            return True
        except sqlite3.IntegrityError as exc:
            LOGGER.warning("Provenance record rejected (duplicate id/nonce): %s", exc)
            return False
        except Exception as exc:
            LOGGER.error("insert_provenance_record failed: %s", exc)
            return False

    def get_provenance_records(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """Return stored provenance records in sequence order.

        Args:
            limit: Maximum rows.

        Returns:
            List of record dictionaries.
        """
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM provenance_records ORDER BY sequence_number ASC LIMIT ?",
                    (int(limit),)).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            LOGGER.error("get_provenance_records failed: %s", exc)
            return []

    def nonce_exists(self, nonce: str) -> bool:
        """Check whether a nonce has already been recorded (replay detection).

        Args:
            nonce: Hex nonce string.

        Returns:
            ``True`` when the nonce is already present.
        """
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM provenance_records WHERE nonce=? LIMIT 1", (nonce,)
                ).fetchone()
            return row is not None
        except Exception as exc:
            LOGGER.error("nonce_exists failed: %s", exc)
            return False

    # -- model registry ----------------------------------------------------
    def register_model(self, model_path: str, weight_hash: str,
                       file_hash: Optional[str] = None, framework: str = "",
                       access_level: str = "", notes: str = "") -> bool:
        """Record a model's identity hashes for later substitution detection.

        Args:
            model_path: Path of the model file.
            weight_hash: Canonical SHA-256 of weights.
            file_hash: SHA-256 of the raw file.
            framework: ``pytorch`` / ``onnx``.
            access_level: Achieved access level.
            notes: Free-form operator note.

        Returns:
            ``True`` on success.
        """
        try:
            with self.connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO model_registry (model_path, weight_hash, file_hash, "
                    "framework, access_level, registered_ns, notes) VALUES (?,?,?,?,?,?,?)",
                    (str(model_path), str(weight_hash), file_hash, framework, access_level,
                     time.time_ns(), notes),
                )
            return True
        except Exception as exc:
            LOGGER.error("register_model failed: %s", exc)
            return False

    def lookup_model(self, weight_hash: str) -> List[Dict[str, Any]]:
        """Find registry entries matching a weight hash.

        Args:
            weight_hash: Canonical weight digest.

        Returns:
            Matching registry rows.
        """
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM model_registry WHERE weight_hash=?", (weight_hash,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            LOGGER.error("lookup_model failed: %s", exc)
            return []

    # -- audit log ---------------------------------------------------------
    def append_audit(self, actor: str, action: str, target: Optional[str] = None,
                     details: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Append a hash-chained entry to the audit log.

        The read-then-insert is atomic: ``connect()`` holds ``self._lock`` for
        the entire ``with`` block, so two concurrent callers will serialize and
        the second caller always reads the first caller's committed hash.

        Args:
            actor: Who performed the action.
            action: Action verb, e.g. ``SCAN_START``.
            target: Object acted upon.
            details: Structured detail payload.

        Returns:
            The new entry hash, or ``None`` on failure.
        """
        try:
            timestamp = time.time_ns()
            detail_json = json.dumps(details or {}, sort_keys=True, default=str)
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
                prev_hash = row["entry_hash"] if row else _GENESIS_HASH
                payload = "|".join([str(timestamp), actor, action, str(target or ""),
                                    detail_json, prev_hash])
                entry_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                conn.execute(
                    "INSERT INTO audit_log (timestamp_ns, actor, action, target, details, "
                    "prev_hash, entry_hash) VALUES (?,?,?,?,?,?,?)",
                    (timestamp, actor, action, target, detail_json, prev_hash, entry_hash),
                )
            return entry_hash
        except Exception as exc:
            LOGGER.error("append_audit failed: %s", exc)
            return None

    def verify_audit_chain(self) -> Dict[str, Any]:
        """Recompute every audit entry hash and validate the chain links.

        Returns:
            Dictionary with ``valid``, ``entries`` and a list of ``breaks``.
        """
        try:
            with self.connect() as conn:
                rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
            prev_hash = _GENESIS_HASH
            breaks: List[Dict[str, Any]] = []
            for row in rows:
                payload = "|".join([
                    str(row["timestamp_ns"]), row["actor"], row["action"],
                    str(row["target"] or ""), row["details"], row["prev_hash"],
                ])
                expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                if row["prev_hash"] != prev_hash:
                    breaks.append({"id": row["id"], "issue": "BROKEN_LINK",
                                   "expected_prev": prev_hash, "found_prev": row["prev_hash"]})
                if expected != row["entry_hash"]:
                    breaks.append({"id": row["id"], "issue": "HASH_MISMATCH",
                                   "expected": expected, "found": row["entry_hash"]})
                prev_hash = row["entry_hash"]
            return {"valid": not breaks, "entries": len(rows), "breaks": breaks,
                    "head_hash": prev_hash}
        except Exception as exc:
            LOGGER.error("verify_audit_chain failed: %s", exc)
            return {"valid": False, "entries": 0, "breaks": [{"issue": str(exc)}],
                    "head_hash": None}

    def get_audit_log(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return recent audit entries, newest first.

        Args:
            limit: Maximum rows.

        Returns:
            Audit rows as dictionaries.
        """
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            LOGGER.error("get_audit_log failed: %s", exc)
            return []

    def audit_trail_hash(self) -> str:
        """Return the head hash of the audit chain for report binding.

        Returns:
            Hex digest of the most recent audit entry (genesis when empty).
        """
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
            return row["entry_hash"] if row else _GENESIS_HASH
        except Exception as exc:
            LOGGER.error("audit_trail_hash failed: %s", exc)
            return _GENESIS_HASH

    def stats(self) -> Dict[str, Any]:
        """Return row counts for every table (dashboard/status use).

        Returns:
            Mapping of table name to row count.
        """
        tables = ("scans", "findings", "contributors", "provenance_records",
                  "model_registry", "audit_log")
        out: Dict[str, Any] = {"database": str(self.path)}
        try:
            with self.connect() as conn:
                for table in tables:
                    if table not in _VALID_TABLES:
                        LOGGER.error("stats: rejected invalid table name '%s'", table)
                        continue
                    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                    out[table] = int(row["n"]) if row else 0
            out["size_bytes"] = self.path.stat().st_size if self.path.is_file() else 0
        except Exception as exc:
            LOGGER.error("stats failed: %s", exc)
            out["error"] = str(exc)
        return out

    def reset(self) -> bool:
        """Delete all rows from every table (demo reset; keeps schema).

        Returns:
            ``True`` on success.
        """
        try:
            with self.connect() as conn:
                for table in ("findings", "contributors", "provenance_records",
                              "model_registry", "audit_log", "scans"):
                    if table not in _VALID_TABLES:
                        LOGGER.error("reset: rejected invalid table name '%s'", table)
                        continue
                    conn.execute(f"DELETE FROM {table}")
            LOGGER.info("Database reset: all rows cleared")
            return True
        except Exception as exc:
            LOGGER.error("reset failed: %s", exc)
            return False


_DB_SINGLETON: Optional[ShieldDatabase] = None
_DB_LOCK = threading.Lock()


def get_database(path: Optional[str | Path] = None) -> ShieldDatabase:
    """Return the process-wide :class:`ShieldDatabase` instance.

    Args:
        path: Optional explicit database path (forces a new instance).

    Returns:
        Shared database handle.
    """
    global _DB_SINGLETON
    with _DB_LOCK:
        if _DB_SINGLETON is None or path is not None:
            _DB_SINGLETON = ShieldDatabase(path)
        return _DB_SINGLETON


__all__ = ["ShieldDatabase", "get_database", "SCHEMA_VERSION"]
