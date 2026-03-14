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

    Orchestrator-only flags (consumed here, not passed through to workers):
        --gpu-workers <N>           (default: 1)
            Give the first N workers GPU access via CUDA_VISIBLE_DEVICES.
            Remaining workers receive CUDA_VISIBLE_DEVICES="" (CPU only).
            Each worker's pipeline script reads this env var and automatically
            overrides its config device fields to match.
        --gpu-ids <IDS>             (default: "0")
            Comma-separated CUDA device IDs assigned to GPU workers.
            Worker i (i < N) receives gpu_ids[i % len(gpu_ids)].
            Example: --gpu-workers 2 --gpu-ids 0,1 for two physical GPU
        --max-worker-memory-gb <GB>
            Hard-kills any worker whose RSS exceeds <GB> gigabytes and logs a
            clear message. Requires psutil (pip install psutil). Useful for
            diagnosing or preventing silent OOM failures.
        --worker-timeout-hours <H>
            Hard-kills any worker still running after <H> hours and records
            its exit code as -1. Useful when a CUDA context hang leaves a
            worker alive but permanently blocked.

Example:
    python src/m_RunParallelPipeline.py --num-workers 2 \\
        --gpu-workers 1 --gpu-ids 0 \\
        --max-worker-memory-gb 20 --worker-timeout-hours 6 \\
        --dataset-prefix MC --limit-datasets 14 \\
        --shapley-eval-budget 270 --optimizer-eval-budget 270 --final-top-k 4

GPU / CPU device assignment:
    By default worker 0 is assigned CUDA_VISIBLE_DEVICES=0 (GPU) and all
    additional workers are assigned CUDA_VISIBLE_DEVICES="" (CPU only). This
    eliminates VRAM contention between workers on a single-GPU machine.

    Use --gpu-workers N to give the first N workers GPU access and the rest
    CPU. Use --gpu-ids to specify which physical GPU IDs are assigned:
        --gpu-workers 2 --gpu-ids 0,1   → W0 gets GPU 0, W1 gets GPU 1
        --gpu-workers 0                 → all workers use CPU

    l_RunMCShapleyOptimizerPipeline.py reads CUDA_VISIBLE_DEVICES at startup
    and automatically overrides the device field in every config it loads to
    match (cuda for GPU workers, cpu for CPU workers). Config YAML files do not
    need to be edited between runs.

CPU / RAM warning:
    Observed steady-state RAM usage scales with --num-workers. With ~40% RAM
    at steady state for a single worker, two workers may approach 80% before
    the OS begins paging. Reduce --num-workers or add swap if memory is tight.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
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
        try:
            master_log.write(item)
            master_log.flush()
        except OSError:
            # Disk full or log file error — continue writing to stdout only so
            # the writer thread keeps draining the queue and does not hang.
            pass


def _memory_watchdog(
    procs: list[tuple[int, subprocess.Popen]],
    limit_bytes: int,
    out_queue: queue.SimpleQueue,
    poll_interval: float = 5.0,
) -> None:
    """Terminate any worker whose RSS exceeds limit_bytes.

    Runs in a daemon thread. Polls every poll_interval seconds and hard-kills
    (SIGKILL / TerminateProcess) any worker whose resident set size exceeds the
    per-worker limit, enqueuing a clear message so the failure is distinguishable
    from a silent OS-level OOM kill in the master log.

    Requires psutil. If psutil is not installed the watchdog logs a warning and
    exits immediately — it does not raise.
    """
    try:
        import psutil
    except ImportError:
        out_queue.put(
            "[PARALLEL] WARNING: psutil not installed — memory watchdog disabled. "
            "Install with: pip install psutil\n"
        )
        return

    limit_gb = limit_bytes / (1024 ** 3)
    active = {i: proc for i, proc in procs}
    while active:
        time.sleep(poll_interval)
        for i, proc in list(active.items()):
            if proc.poll() is not None:
                del active[i]
                continue
            try:
                rss = psutil.Process(proc.pid).memory_info().rss
                if rss > limit_bytes:
                    proc.kill()  # hard kill; terminate() may be ignored on Windows
                    rss_gb = rss / (1024 ** 3)
                    out_queue.put(
                        f"[PARALLEL] Worker {i} KILLED by memory watchdog: "
                        f"RSS {rss_gb:.2f} GB exceeded per-worker limit {limit_gb:.2f} GB\n"
                    )
                    del active[i]
            except Exception:  # process gone, access denied, etc.
                del active[i]


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
    parser.add_argument(
        "--max-worker-memory-gb",
        type=float,
        default=None,
        metavar="GB",
        help=(
            "Per-worker RAM limit in GB. If a worker's resident set size exceeds "
            "this value it is hard-killed and a clear message is written to the "
            "master log. Requires psutil (pip install psutil). "
            "No limit by default."
        ),
    )
    parser.add_argument(
        "--worker-timeout-hours",
        type=float,
        default=None,
        metavar="H",
        help=(
            "Per-worker wall-clock timeout in hours. Any worker still running "
            "after this deadline is hard-killed and its exit code recorded as -1. "
            "Useful when a CUDA context hang leaves a worker alive but permanently "
            "blocked. No timeout by default."
        ),
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of workers that receive GPU access via CUDA_VISIBLE_DEVICES "
            "(default: 1). Workers 0..N-1 are assigned the IDs from --gpu-ids; "
            "workers N..num-workers-1 get CUDA_VISIBLE_DEVICES='' (CPU only). "
            "Pass 0 to force all workers to CPU."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        metavar="IDS",
        help=(
            "Comma-separated CUDA device IDs to assign to GPU workers "
            "(default: '0'). Worker i (i < --gpu-workers) receives "
            "gpu_ids[i %% len(gpu_ids)], so IDs are cycled if there are fewer "
            "IDs than GPU workers. Example: --gpu-workers 2 --gpu-ids 0,1"
        ),
    )
    args, passthrough = parser.parse_known_args()
    num_workers = max(1, int(args.num_workers))
    max_worker_memory_bytes: int | None = (
        int(args.max_worker_memory_gb * (1024 ** 3))
        if args.max_worker_memory_gb is not None
        else None
    )
    worker_timeout_seconds: float | None = (
        args.worker_timeout_hours * 3600.0
        if args.worker_timeout_hours is not None
        else None
    )
    num_gpu_workers: int = max(0, min(int(args.gpu_workers), num_workers))
    gpu_ids: list[str] = [s.strip() for s in args.gpu_ids.split(",") if s.strip()] or ["0"]

    def _worker_cvd(i: int) -> str:
        """Return the CUDA_VISIBLE_DEVICES value for worker i."""
        return gpu_ids[i % len(gpu_ids)] if i < num_gpu_workers else ""

    if num_workers == 1:
        # Single worker: run the pipeline directly without subprocess overhead.
        cmd = [sys.executable, str(_PIPELINE_SCRIPT)] + passthrough
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": _worker_cvd(0)}
        return subprocess.run(cmd, env=env).returncode

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
        device_summary = "  ".join(
            f"W{i}→{'CUDA:' + _worker_cvd(i) if i < num_gpu_workers else 'CPU'}"
            for i in range(num_workers)
        )
        _log(f"[PARALLEL] Devices    : {num_gpu_workers}/{num_workers} GPU — {device_summary}")
        if max_worker_memory_bytes is not None:
            _log(f"[PARALLEL] RAM limit  : {args.max_worker_memory_gb:.2f} GB per worker (watchdog active)")
        else:
            _log(f"[PARALLEL] RAM limit  : none (pass --max-worker-memory-gb to enable watchdog)")
        if worker_timeout_seconds is not None:
            _log(f"[PARALLEL] Timeout    : {args.worker_timeout_hours:.2f} h per worker")
        else:
            _log(f"[PARALLEL] Timeout    : none (pass --worker-timeout-hours to enable)")
        _log("")

        out_queue: queue.SimpleQueue = queue.SimpleQueue()
        procs: list[tuple[int, subprocess.Popen]] = []
        collector_threads: list[tuple[threading.Thread, threading.Event]] = []

        for i in range(num_workers):
            cmd = [sys.executable, str(_PIPELINE_SCRIPT)] + passthrough + [
                "--num-workers", str(num_workers),
                "--worker-index", str(i),
            ]
            cvd = _worker_cvd(i)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": cvd}
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout pipe
                env=env,
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
            device_label = f"CUDA:{cvd}" if cvd else "CPU"
            _log(f"[PARALLEL] Worker {i} started (pid={proc.pid}, device={device_label})")

        _log("")

        # Start memory watchdog if a per-worker limit was requested.
        if max_worker_memory_bytes is not None:
            watchdog = threading.Thread(
                target=_memory_watchdog,
                args=(procs, max_worker_memory_bytes, out_queue),
                daemon=True,  # daemon: exits automatically when all workers are done
            )
            watchdog.start()

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
            try:
                results[i] = proc.wait(timeout=worker_timeout_seconds)
            except subprocess.TimeoutExpired:
                out_queue.put(
                    f"[PARALLEL] Worker {i} KILLED: exceeded timeout of "
                    f"{worker_timeout_seconds / 3600:.1f} h\n"
                )
                proc.kill()
                results[i] = -1

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
