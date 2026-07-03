#!/usr/bin/env python3
"""CLI for the unpublishable Archetype V2 diagnostic map preview."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from story_monitor.archetype_preview_v2 import export_archetype_preview


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a causal Archetype V2 diagnostic viewer generation."
    )
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-failed-quality-gates",
        action="store_true",
        help="Create an explicitly unpublishable diagnostic preview when hard gates pass.",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit")
    parser.add_argument("--temp-dir")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = export_archetype_preview(
        Path(args.generation_dir),
        Path(args.model_dir),
        Path(args.evaluation_dir),
        Path(args.output_dir),
        allow_failed_quality_gates=args.allow_failed_quality_gates,
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
