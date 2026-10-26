from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data/research/generation_checkpoint_v1_1.json"
DEFAULT_STDOUT_LOG = PROJECT_ROOT / "data/research/v1_1_generation.stdout.log"
DEFAULT_STDERR_LOG = PROJECT_ROOT / "data/research/v1_1_generation.stderr.log"
DEFAULT_SUPERVISOR_LOG = PROJECT_ROOT / "data/research/v1_1_supervisor.log"
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_checkpoint(path: Path, attempts: int = 5) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            if attempt < attempts:
                time.sleep(0.2 * attempt)
    return {}


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _keep_system_awake() -> bool:
    if os.name != "nt":
        return False
    previous_state = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )
    return bool(previous_state)


def _event(log_path: Path, event: str, **details: Any) -> None:
    record = {"at": _utc_now(), "event": event, **details}
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _last_error_text(state: dict[str, Any]) -> str:
    error = state.get("last_error")
    if isinstance(error, dict):
        return f"{error.get('type', '')}: {error.get('message', '')}".strip()
    return str(error or "")


def _retry_delay(state: dict[str, Any], normal_delay: int, pool_delay: int) -> int:
    error = _last_error_text(state).lower()
    if any(
        marker in error
        for marker in (
            "insufficientproviderpoolerror",
            "fewer_than_four_available_providers",
            "only 3 configured providers remain",
            "only 2 configured providers remain",
            "only 1 configured providers remain",
        )
    ):
        return pool_delay
    return normal_delay


def _wait_for_attached_process(
    pid: int,
    poll_seconds: int,
    supervisor_log: Path,
) -> None:
    if not _process_exists(pid):
        _event(supervisor_log, "attached_process_not_running", pid=pid)
        return
    _event(supervisor_log, "attached_to_running_generator", pid=pid)
    while _process_exists(pid):
        time.sleep(poll_seconds)
    _event(supervisor_log, "attached_generator_exited", pid=pid)


def supervise(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.resolve()
    stdout_log = args.stdout_log.resolve()
    stderr_log = args.stderr_log.resolve()
    supervisor_log = args.supervisor_log.resolve()
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    _event(
        supervisor_log,
        "supervisor_started",
        target_sets=args.target_sets,
        system_sleep_prevented=_keep_system_awake(),
    )

    if args.attach_pid:
        _wait_for_attached_process(
            args.attach_pid,
            args.poll_seconds,
            supervisor_log,
        )

    restart_count = 0
    while True:
        state = _read_checkpoint(checkpoint)
        completed_sets = int(state.get("completed_sets") or 0)
        if completed_sets >= args.target_sets:
            _event(
                supervisor_log,
                "target_completed",
                completed_sets=completed_sets,
                target_sets=args.target_sets,
                restart_count=restart_count,
            )
            return 0

        if restart_count:
            delay = _retry_delay(
                state,
                args.retry_delay_seconds,
                args.provider_pool_retry_delay_seconds,
            )
            _event(
                supervisor_log,
                "waiting_before_restart",
                delay_seconds=delay,
                completed_sets=completed_sets,
                last_error=_last_error_text(state),
            )
            time.sleep(delay)

        restart_count += 1
        command = [
            sys.executable,
            "-u",
            "-B",
            "scripts/generate_weak_supervision_dataset.py",
            "--target-sets",
            str(args.target_sets),
            "--checkpoint",
            str(checkpoint),
        ]
        _event(
            supervisor_log,
            "starting_generator",
            restart_count=restart_count,
            completed_sets=completed_sets,
            command=command,
        )
        with (
            stdout_log.open("a", encoding="utf-8") as stdout_handle,
            stderr_log.open("a", encoding="utf-8") as stderr_handle,
        ):
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )

        state = _read_checkpoint(checkpoint)
        _event(
            supervisor_log,
            "generator_exited",
            exit_code=result.returncode,
            completed_sets=int(state.get("completed_sets") or 0),
            status=state.get("status"),
            last_error=_last_error_text(state),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the resumable V1.1 dataset generator running until its target is complete."
    )
    parser.add_argument("--target-sets", type=int, default=1000)
    parser.add_argument("--attach-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--retry-delay-seconds", type=int, default=60)
    parser.add_argument("--provider-pool-retry-delay-seconds", type=int, default=900)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stdout-log", type=Path, default=DEFAULT_STDOUT_LOG)
    parser.add_argument("--stderr-log", type=Path, default=DEFAULT_STDERR_LOG)
    parser.add_argument("--supervisor-log", type=Path, default=DEFAULT_SUPERVISOR_LOG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_sets <= 0 or args.target_sets % 4 != 0:
        raise ValueError("target_sets must be positive and divisible by four")
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
