#!/usr/bin/env python3
"""CLI for the observation/video comparison pipeline.

Examples (from the repository root):

  PYTHONPATH=continuum_filament_model/src python continuum_filament_model/video_compare.py \
      extract --video img/gray5.mp4 --output /tmp/growing-string-gray5

  PYTHONPATH=continuum_filament_model/src python continuum_filament_model/video_compare.py \
      all --video img/gray5.mp4 --output /tmp/growing-string-gray5 \
      --model /tmp/model/trajectory.npz --registration /tmp/registration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from growing_filament.video_comparison import (  # noqa: E402
    RegistrationConfig,
    SegmentationConfig,
    compare_with_model,
    render_comparison,
    run_pipeline,
)


def _json_file(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config(args: argparse.Namespace) -> SegmentationConfig:
    values = _json_file(args.config)
    # CLI values intentionally override the JSON file only when supplied.
    for name in ("polarity", "background", "contrast", "threshold"):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    for name in ("frame_stride", "min_component_size", "max_components", "max_centerline_points"):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if getattr(args, "roi", None) is not None:
        values["roi"] = args.roi
    return SegmentationConfig.from_mapping(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "compare", "render", "all"):
        child = sub.add_parser(command)
        child.add_argument("--video", required=command in {"extract", "render", "all"})
        child.add_argument("--output", required=True, help="analysis/artifact directory")
        child.add_argument("--model", help="trajectory .npz, model centerline CSV, or JSON")
        child.add_argument("--config", help="segmentation JSON")
        child.add_argument("--registration", help="registration JSON; omit to suppress metrics")
        child.add_argument("--filament-id")
        child.add_argument("--max-frames", type=int)
        child.add_argument("--representative-count", type=int, default=6)
        if command in {"extract", "all"}:
            child.add_argument("--polarity", choices=("dark", "bright"))
            child.add_argument("--background", choices=("none", "median", "local_median", "scalar"))
            child.add_argument("--contrast", choices=("none", "percentile"))
            child.add_argument("--threshold", choices=("otsu", "absolute", "percentile"))
            child.add_argument("--frame-stride", type=int)
            child.add_argument("--min-component-size", type=int)
            child.add_argument("--max-components", type=int)
            child.add_argument("--max-centerline-points", type=int)
            child.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    config = _config(args) if args.command in {"extract", "all"} else SegmentationConfig()
    registration = RegistrationConfig.from_mapping(_json_file(args.registration))
    if args.command in {"extract", "all"}:
        if not args.video:
            raise SystemExit("--video is required")
        run_pipeline(args.video, output, config, max_frames=args.max_frames)
    if args.command in {"compare", "all"}:
        if not args.model:
            raise SystemExit("--model is required for compare/all")
        compare_with_model(output, args.model, registration, filament_id=args.filament_id, output_dir=output)
    if args.command in {"render", "all"}:
        if not args.video or not args.model:
            raise SystemExit("--video and --model are required for render/all")
        render_comparison(
            args.video,
            output,
            args.model,
            output,
            registration,
            max_video_frames=args.max_frames,
            representative_count=args.representative_count,
        )
    print(json.dumps({"output": str(output.resolve()), "command": args.command}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
