"""Docker-native runner for tests owned by langgraph-runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import argparse


def main() -> int:
    if os.environ.get("RUN_RUNTIME_DB_MIGRATIONS", "false").lower() == "true":
        subprocess.run([sys.executable, "-m", "langgraph_runtime.migrate"], check=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    target = os.environ.get("RUNTIME_TEST_TARGET", "/app/langgraph_runtime/tests")
    if args.test:
        target = f"{target}::{args.test}"
    command = ["pytest", target]
    if args.verbose:
        command.append("-v")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
