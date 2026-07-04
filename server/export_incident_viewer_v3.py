from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_monitor.incident_viewer_v3 import export_incident_viewer_v3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an immutable crop-incident V3 story-map viewer bundle."
    )
    parser.add_argument("--incident-dir", required=True, type=Path)
    parser.add_argument("--source-generation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit")
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--min-valid-geometry-coverage", type=float, default=0.95)
    parser.add_argument("--min-frame-geometry-coverage", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_incident_viewer_v3(
        args.incident_dir,
        args.source_generation_dir,
        args.output_dir,
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
        min_valid_geometry_coverage=args.min_valid_geometry_coverage,
        min_frame_geometry_coverage=args.min_frame_geometry_coverage,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
