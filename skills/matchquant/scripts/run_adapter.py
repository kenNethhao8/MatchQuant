#!/usr/bin/env python3
"""Helper script to invoke MatchQuant agent adapter with JSON string or file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add worktree to sys.path if not installed in site-packages
worktree_root = Path(__file__).resolve().parent.parent.parent
if str(worktree_root) not in sys.path:
    sys.path.insert(0, str(worktree_root))

from matchquant.adapter import execute_adapter_request


def main() -> int:
    parser = argparse.ArgumentParser(description="MatchQuant Agent Adapter CLI Invoker")
    parser.add_argument("--request", help="JSON string representing adapter request")
    parser.add_argument("--file", help="Path to JSON file representing adapter request")
    args = parser.parse_args()

    payload: dict = {}
    if args.file:
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    elif args.request:
        payload = json.loads(args.request)
    else:
        if not sys.stdin.isatty():
            content = sys.stdin.read().strip()
            if content:
                payload = json.loads(content)

    if not payload:
        parser.error("must supply --request, --file, or piped JSON on stdin")

    res = execute_adapter_request(payload)
    # Output clean JSON to stdout
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res.get("status") in ("ok", "needs_review", "applied", "dry_run_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
