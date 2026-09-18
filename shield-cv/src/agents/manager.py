"""Office manager: discovers contributors and runs one agent per contributor.

The manager is the orchestration half of the multi-agent office. It walks a
dataset root, decides which sub-folders represent distinct contributors, and
dispatches one :func:`src.agents.agent.agent_scan_contributor` call per
contributor across a :class:`concurrent.futures.ProcessPoolExecutor`.

Processes, not threads, are used because the detectors are CPU-bound numeric
work that holds the GIL. Worker count is capped both by configuration and by
the actual core count, because the target hardware is a 4-core i5 with 8 GB of
RAM and each worker loads its own copy of the embedding backbone. Oversub-
scribing that machine causes swapping, which is far slower than running
sequentially.

A single-worker configuration runs in-process instead of paying for a pool,
which also makes the whole pipeline debuggable and safe to call from contexts
where re-entrant process spawning is not allowed.
"""

from __future__ import annotations

import fnmatch
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agents.agent import AgentReport, agent_scan_dict
from src.config import get_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Folder names that are part of a dataset's internal layout rather than
# separate contributors. Without this guard a plain `images/` + `labels/`
# dataset would be mistaken for two contributors.
_STRUCTURAL_DIRS = {
    "images", "labels", "annotations", "train", "val", "valid", "test",
    "__pycache__", ".git", ".ipynb_checkpoints",
}


class OfficeManager:
    """Discovers contributor folders and runs a parallel agent scan over them.

    Attributes:
        cfg: Loaded configuration object.
        max_workers: Effective worker cap for the process pool.
        timeout: Per-agent wall-clock timeout in seconds.
        patterns: Glob patterns identifying contributor folders.
    """

    def __init__(self, max_workers: Optional[int] = None,
                 timeout: Optional[float] = None) -> None:
        """Initialise the manager from configuration.

        Args:
            max_workers: Override for the configured worker cap.
            timeout: Override for the configured per-agent timeout in seconds.
        """
        self.cfg = get_config()
        try:
            section = self.cfg.section("agents") or {}
        except Exception as exc:
            LOGGER.error("agents config unavailable, using defaults: %s", exc)
            section = {}

        configured = int(section.get("max_workers", 4) or 4)
        if max_workers is not None:
            configured = int(max_workers)
        cores = os.cpu_count() or 1

        # Never spawn more workers than cores: each worker holds its own
        # backbone, and oversubscription means swapping.
        limit = min(configured, cores)

        # Cores are not the binding constraint — memory is. Every worker loads
        # its own torch runtime plus a ResNet-18 backbone, roughly 700 MB
        # resident. Sizing purely by core count gets workers OOM-killed, which
        # surfaces as "a process in the process pool was terminated abruptly"
        # and loses the whole batch. Budget by available RAM instead.
        self.worker_memory_mb = float(section.get("worker_memory_mb", 700) or 700)
        available = self._available_memory_mb()
        if available is not None:
            affordable = max(1, int(available // max(self.worker_memory_mb, 1.0)))
            if affordable < limit:
                LOGGER.info("office: capping workers %d -> %d (%.0f MB available, "
                            "~%.0f MB per worker)", limit, affordable, available,
                            self.worker_memory_mb)
            limit = min(limit, affordable)
        self.available_memory_mb = available
        self.max_workers = max(1, limit)

        self.timeout = float(timeout if timeout is not None
                             else section.get("per_agent_timeout_s", 1800) or 1800)
        self.patterns: List[str] = list(
            section.get("contributor_dir_patterns")
            or ["contributor_*", "source_*", "vendor_*"])

    @staticmethod
    def _available_memory_mb() -> Optional[float]:
        """Report memory actually available for new worker processes.

        ``MemAvailable`` is used in preference to ``MemFree`` because reclaimable
        page cache is genuinely usable. Returns ``None`` on platforms where this
        cannot be determined, in which case the caller falls back to the core
        count alone.

        Returns:
            Available memory in megabytes, or ``None`` if unknown.
        """
        try:
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                for line in meminfo.read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / 1024.0
            # Windows and macOS: sysconf is unavailable or reports total only,
            # so fall back to total physical memory with a conservative margin.
            pages = os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") else 0
            page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 0
            if pages and page_size:
                return float(pages) * float(page_size) / (1024.0 * 1024.0)
        except Exception as exc:
            LOGGER.debug("available memory probe failed: %s", exc)
        return None

    def discover(self, root: str | Path) -> List[Dict[str, str]]:
        """Find the contributor folders under a dataset root.

        Discovery runs in three passes, most specific first: folders matching a
        configured contributor pattern; otherwise any non-structural
        sub-folders; otherwise the root itself treated as a single contributor.

        Args:
            root: Dataset root directory.

        Returns:
            List of ``{"contributor": name, "path": path}`` dictionaries,
            sorted by contributor name. Empty when the root does not exist.
        """
        try:
            base = Path(root)
            if not base.is_dir():
                LOGGER.error("discover: not a directory: %s", base)
                return []

            children = sorted(p for p in base.iterdir() if p.is_dir())

            matched = [p for p in children
                       if any(fnmatch.fnmatch(p.name, pat) for pat in self.patterns)]
            if matched:
                return [{"contributor": p.name, "path": str(p)} for p in matched]

            generic = [p for p in children if p.name.lower() not in _STRUCTURAL_DIRS]
            if generic:
                LOGGER.info("discover: no pattern match under %s; treating %d "
                            "sub-folder(s) as contributors", base, len(generic))
                return [{"contributor": p.name, "path": str(p)} for p in generic]

            LOGGER.info("discover: %s has no contributor sub-folders; "
                        "scanning it as a single contributor", base)
            return [{"contributor": base.name, "path": str(base)}]

        except Exception as exc:
            LOGGER.error("discover failed for %s: %s", root, exc)
            return []

    def run(self, root: str | Path,
            contributors: Optional[Sequence[str]] = None,
            max_images: Optional[int] = None,
            parallel: bool = True,
            progress_callback: Optional[Callable[[str, int, int], None]] = None
            ) -> Dict[str, Any]:
        """Run one agent per discovered contributor and collect their reports.

        Args:
            root: Dataset root containing contributor folders.
            contributors: Optional subset of contributor names to scan.
            max_images: Optional per-contributor image cap for fast runs.
            parallel: Whether to use a process pool. Forced off when only one
                contributor or one worker is in play.
            progress_callback: Optional ``(stage, done, total)`` callback fired
                as each agent completes.

        Returns:
            Dictionary with ``module``, ``root``, ``reports``, ``agents``,
            ``errors``, ``workers``, ``limitations`` and ``duration_seconds``.
        """
        started = time.time()
        result: Dict[str, Any] = {
            "module": "office_manager",
            "root": str(root),
            "reports": [],
            "agents": 0,
            "errors": [],
            "workers": 1,
            "parallel": False,
            "limitations": [],
            "duration_seconds": 0.0,
        }

        try:
            tasks = self.discover(root)
            if contributors:
                wanted = {str(c) for c in contributors}
                tasks = [t for t in tasks if t["contributor"] in wanted]
                missing = wanted - {t["contributor"] for t in tasks}
                for name in sorted(missing):
                    result["errors"].append(f"contributor not found: {name}")

            if not tasks:
                result["limitations"].append(
                    f"No contributor folders discovered under '{root}'.")
                result["duration_seconds"] = round(time.time() - started, 3)
                return result

            for index, task in enumerate(tasks, start=1):
                task["agent_id"] = f"AGENT-{index:02d}"
                task["max_images"] = max_images

            workers = min(self.max_workers, len(tasks))
            use_pool = bool(parallel) and workers > 1
            result["agents"] = len(tasks)
            result["workers"] = workers if use_pool else 1
            result["parallel"] = use_pool

            LOGGER.info("office: %d contributor(s) under %s, %s",
                        len(tasks), root,
                        f"{workers} worker(s)" if use_pool else "sequential")

            reports = (self._run_pool(tasks, workers, progress_callback)
                       if use_pool
                       else self._run_serial(tasks, progress_callback))

            # Stable ordering: riskiest first, so the CLI table leads with the
            # contributor that needs attention.
            reports.sort(key=lambda r: (-float(r.get("risk_score", 0.0) or 0.0),
                                        str(r.get("contributor", ""))))
            result["reports"] = reports

            for report in reports:
                if report.get("status") == "ERROR":
                    result["errors"].append(
                        f"{report.get('contributor', '?')}: {report.get('error', 'unknown error')}")
                for limitation in report.get("limitations") or []:
                    entry = f"[{report.get('contributor', '?')}] {limitation}"
                    if entry not in result["limitations"]:
                        result["limitations"].append(entry)

        except Exception as exc:
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            LOGGER.error("office run failed: %s\n%s", exc, traceback.format_exc())

        result["duration_seconds"] = round(time.time() - started, 3)
        return result

    def _run_serial(self, tasks: List[Dict[str, Any]],
                    progress_callback: Optional[Callable[[str, int, int], None]]
                    ) -> List[Dict[str, Any]]:
        """Run every agent in the current process, one after another.

        Args:
            tasks: Prepared agent task dictionaries.
            progress_callback: Optional progress callback.

        Returns:
            List of agent report dictionaries.
        """
        reports: List[Dict[str, Any]] = []
        total = len(tasks)
        for done, task in enumerate(tasks, start=1):
            try:
                reports.append(agent_scan_dict(task))
            except Exception as exc:  # pragma: no cover - agent is defensive
                LOGGER.error("serial agent failed: %s", exc)
                reports.append(AgentReport(
                    agent_id=str(task.get("agent_id", "")),
                    contributor=str(task.get("contributor", "")),
                    path=str(task.get("path", "")),
                    status="ERROR", error=str(exc)).to_dict())
            self._notify(progress_callback, task.get("contributor", ""), done, total)
        return reports

    def _run_pool(self, tasks: List[Dict[str, Any]], workers: int,
                  progress_callback: Optional[Callable[[str, int, int], None]]
                  ) -> List[Dict[str, Any]]:
        """Run agents across a process pool, falling back to serial on failure.

        A pool can fail to start outright in restricted environments (no
        ``/dev/shm``, spawn restrictions, frozen executables). That is an
        infrastructure problem, not an analysis result, so the whole batch is
        retried serially rather than reported as scan errors.

        Args:
            tasks: Prepared agent task dictionaries.
            workers: Number of worker processes.
            progress_callback: Optional progress callback.

        Returns:
            List of agent report dictionaries.
        """
        reports: List[Dict[str, Any]] = []
        total = len(tasks)
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(agent_scan_dict, task): task for task in tasks}
                for future in as_completed(futures, timeout=self.timeout * max(1, total)):
                    task = futures[future]
                    try:
                        reports.append(future.result(timeout=self.timeout))
                    except FutureTimeout:
                        LOGGER.error("agent timed out on %s", task.get("contributor"))
                        reports.append(AgentReport(
                            agent_id=str(task.get("agent_id", "")),
                            contributor=str(task.get("contributor", "")),
                            path=str(task.get("path", "")),
                            status="ERROR",
                            error=f"agent exceeded {self.timeout:.0f}s timeout").to_dict())
                    except BrokenProcessPool as exc:
                        # A worker was killed (almost always the OOM killer).
                        # The remaining futures are unusable, so abandon the
                        # pool and redo the whole batch in-process rather than
                        # reporting phantom "errors" for real contributors.
                        LOGGER.error("worker process died (%s); falling back to "
                                     "serial execution for all contributors", exc)
                        raise
                    except Exception as exc:
                        LOGGER.error("agent crashed on %s: %s",
                                     task.get("contributor"), exc)
                        reports.append(AgentReport(
                            agent_id=str(task.get("agent_id", "")),
                            contributor=str(task.get("contributor", "")),
                            path=str(task.get("path", "")),
                            status="ERROR", error=str(exc)).to_dict())
                    done += 1
                    self._notify(progress_callback,
                                 str(task.get("contributor", "")), done, total)
            return reports
        except Exception as exc:
            LOGGER.error("process pool unavailable (%s); falling back to serial", exc)
            return self._run_serial(tasks, progress_callback)

    @staticmethod
    def _notify(callback: Optional[Callable[[str, int, int], None]],
                stage: str, done: int, total: int) -> None:
        """Invoke a progress callback without letting it break the run.

        Args:
            callback: Callback to invoke, or ``None``.
            stage: Stage label, normally the contributor name.
            done: Completed unit count.
            total: Total unit count.
        """
        if callback is None:
            return
        try:
            callback(stage, done, total)
        except Exception as exc:  # pragma: no cover - user callback
            LOGGER.error("progress callback failed: %s", exc)


def run_office(root: str | Path,
               contributors: Optional[Sequence[str]] = None,
               max_images: Optional[int] = None,
               max_workers: Optional[int] = None,
               parallel: bool = True,
               progress_callback: Optional[Callable[[str, int, int], None]] = None
               ) -> Dict[str, Any]:
    """Convenience wrapper running a full multi-agent office scan.

    Args:
        root: Dataset root containing contributor folders.
        contributors: Optional subset of contributor names.
        max_images: Optional per-contributor image cap.
        max_workers: Optional worker cap override.
        parallel: Whether to use a process pool.
        progress_callback: Optional ``(stage, done, total)`` callback.

    Returns:
        Office result dictionary as described in :meth:`OfficeManager.run`.
    """
    return OfficeManager(max_workers=max_workers).run(
        root, contributors=contributors, max_images=max_images,
        parallel=parallel, progress_callback=progress_callback)


__all__ = ["OfficeManager", "run_office"]
