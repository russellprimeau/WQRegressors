"""
Parallel orchestrator for l_RunMCShapleyOptimizerPipeline.py.

Partitions the discovered datasets across N worker processes using stride
assignment: worker i processes plans[i::N]. All workers run concurrently.

Output and logging:
- All workers' stdout/stderr is merged into a single master log file created
    by the orchestrator (e.g. parallel_pipeline_TIMESTAMP.log). Each line is
    prefixed with [W{i}] so output from different workers is distinguishable.
- Each worker also writes its own individual log file (existing behaviour of
    l_RunMCShapleyOptimizerPipeline.py). The individual log path is printed by
    the worker itself near the start of its output.
- The orchestrator prints the master log path at startup. Monitor it with:
        Get-Content "<master_log_path>" -Wait       (PowerShell)
        tail -f "<master_log_path>"                 (bash/WSL)

Output routing design:
- Each worker's stdout/stderr pipe is read by a dedicated collector thread.
    Collector threads only put lines into a shared in-memory queue — they
    never perform I/O themselves and never block on locks.
- A single writer thread drains the queue and does all terminal/file I/O
    sequentially. This avoids the deadlock that occurs when a shared lock is
    held during I/O: a slow write would block one collector thread from
    draining its pipe, filling the pipe buffer and stalling the worker process.

Usage:
    python src/m_RunParallelPipeline.py --num-workers 2 [all pipeline args]

Example:
    python src/m_RunParallelPipeline.py --num-workers 2 \\
        --dataset-prefix MC --limit-datasets 14 \\
        --shapley-eval-budget 270 --optimizer-eval-budget 270 --final-top-k 4

GPU memory warning:
    Each worker process runs its own CUDA context and CuPy memory pool. Peak
    VRAM usage scales approximately as N × single-worker peak. Monitor VRAM
    during the first parallel run and reduce --num-workers (or set device: cpu
    in config files) if GPU OOM errors occur.

CPU / RAM warning:
    Observed steady-state RAM usage scales with --num-workers. With ~40% RAM
    at steady state for a single worker, two workers may approach 80% before
    the OS begins paging. Reduce --num-workers or add swap if memory is tight.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

_PIPELINE_SCRIPT = Path(__file__).resolve().parent / "l_RunMCShapleyOptimizerPipeline.py"
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

_SENTINEL = object()  # signals that a collector thread has finished


def _resolve_log_dir(passthrough: list[str]) -> Path:
    """Parse --data-root from passthrough args to locate the pipeline_logs dir."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data-root", default="data/output/regression")
    known, _ = p.parse_known_args(passthrough)
    data_root = Path(known.data_root)
    if not data_root.is_absolute():
        data_root = (_WORKSPACE_ROOT / data_root).resolve()
    return (data_root.parent / "pipeline_logs").resolve()


def _collect_worker_output(
    stream,
    worker_index: int,
    out_queue: queue.SimpleQueue,
    done: threading.Event,
) -> None:
    """Read lines from a worker pipe and enqueue them for the writer thread.

    This thread never performs terminal or file I/O — it only puts lines into
    the in-memory queue. This prevents the collector from blocking on I/O and
    stalling the pipe, which would cause the worker process to deadlock.
    """
    prefix = f"[W{worker_index}] "
    try:
        for raw in stream:
            line = raw.decode("utf-8", errors="replace")
            out_queue.put(prefix + line)
    except Exception as exc:
        out_queue.put(f"{prefix}[COLLECTOR-ERROR] {type(exc).__name__}: {exc}\n")
    finally:
        done.set()
        out_queue.put(_SENTINEL)  # signal writer that this collector is done


def _write_output(
    out_queue: queue.SimpleQueue,
    num_collectors: int,
    master_log,
) -> None:
    """Drain the output queue and write each line to stdout and the master log.

    Runs in a single dedicated thread so all I/O is sequential — no locking
    required. Exits after receiving one sentinel per collector thread.
    """
    completed = 0
    while completed < num_collectors:
        item = out_queue.get()
        if item is _SENTINEL:
            completed += 1
            continue
        sys.stdout.write(item)
        sys.stdout.flush()
        master_log.write(item)
        master_log.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel orchestrator for l_RunMCShapleyOptimizerPipeline.py. "
            "Partitions datasets across --num-workers concurrent processes. "
            "All other arguments are passed through to each worker unchanged. "
            "All workers' output is merged into a single master log file."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel worker processes (default: 2).",
    )
    args, passthrough = parser.parse_known_args()
    num_workers = max(1, int(args.num_workers))

    if num_workers == 1:
        # Single worker: run the pipeline directly without subprocess overhead.
        cmd = [sys.executable, str(_PIPELINE_SCRIPT)] + passthrough
        return subprocess.run(cmd).returncode

    log_dir = _resolve_log_dir(passthrough)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    master_log_path = log_dir / f"parallel_pipeline_{timestamp}.log"

    with open(master_log_path, "w", encoding="utf-8", buffering=1) as master_log:

        def _log(msg: str) -> None:
            """Write an orchestrator status line to stdout and the master log."""
            line = msg + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()
            master_log.write(line)
            master_log.flush()

        _log(f"[PARALLEL] Master log : {master_log_path}")
        _log(f"[PARALLEL] Monitor    : Get-Content \"{master_log_path}\" -Wait")
        _log(f"[PARALLEL] Workers    : {num_workers}")
        _log(f"[PARALLEL] Partition  : stride 1/{num_workers} — worker i runs plans[i::{num_workers}]")
        _log(f"[PARALLEL] GPU note   : peak VRAM scales with number of workers using GPU simultaneously")
        _log("")

        out_queue: queue.SimpleQueue = queue.SimpleQueue()
        procs: list[tuple[int, subprocess.Popen]] = []
        collector_threads: list[tuple[threading.Thread, threading.Event]] = []

        for i in range(num_workers):
            cmd = [sys.executable, str(_PIPELINE_SCRIPT)] + passthrough + [
                "--num-workers", str(num_workers),
                "--worker-index", str(i),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout pipe
            )
            done = threading.Event()
            t = threading.Thread(
                target=_collect_worker_output,
                args=(proc.stdout, i, out_queue, done),
                daemon=True,
            )
            t.start()
            procs.append((i, proc))
            collector_threads.append((t, done))
            _log(f"[PARALLEL] Worker {i} started (pid={proc.pid})")

        _log("")

        # Writer thread drains the queue; runs until all collectors send sentinels.
        writer = threading.Thread(
            target=_write_output,
            args=(out_queue, num_workers, master_log),
            daemon=False,  # non-daemon: must finish draining before process exits
        )
        writer.start()

        # Wait for all worker processes to exit, then wait for the writer to drain.
        results: dict[int, int] = {}

        def _wait_for_worker(i: int, proc: subprocess.Popen) -> None:
            results[i] = proc.wait()

        wait_threads = [
            threading.Thread(target=_wait_for_worker, args=(i, proc), daemon=True)
            for i, proc in procs
        ]
        for t in wait_threads:
            t.start()
        for t in wait_threads:
            t.join()

        writer.join()  # drain any remaining lines after all workers exit

        _log("")
        failed = [i for i, rc in results.items() if rc != 0]
        for i, rc in sorted(results.items()):
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            _log(f"[PARALLEL] Worker {i}: {status}")

        if failed:
            _log(f"[PARALLEL] {len(failed)}/{num_workers} worker(s) failed: {failed}")
            return 2

        _log(f"[PARALLEL] All {num_workers} workers completed successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
