"""Command-line interface for inference, evaluation, and benchmarking."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import load_config
from .evaluation.benchmark import benchmark_video
from .evaluation.detection import evaluate_detection
from .evaluation.tracking import evaluate_tracking
from .exceptions import PerceptionEngineError
from .logging_utils import configure_logging
from .pipeline.runner import run_video

LOGGER = logging.getLogger(__name__)


def _source(value: str) -> str | int:
    return int(value) if value.isdecimal() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ape",
        description="YOLO11 + Kalman + Hungarian autonomous-driving perception pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Process and annotate a video")
    run.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    run.add_argument("--source", required=True, help="Video path, stream URL, or camera index")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mot-output", type=Path, help="Optional MOTChallenge prediction file")
    run.add_argument("--display", action="store_true", help="Show live output; press q to stop")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--preserve-audio", action="store_true")
    run.add_argument("--device", help="YOLO device override, e.g. cpu, 0, or cuda:0")

    detection = subparsers.add_parser(
        "evaluate-detection", help="Evaluate YOLO on a labeled dataset"
    )
    detection.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    detection.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    detection.add_argument("--output", type=Path, required=True)
    detection.add_argument("--split", default="val")

    tracking = subparsers.add_parser(
        "evaluate-tracking", help="Evaluate MOTChallenge predictions against identity ground truth"
    )
    tracking.add_argument("--ground-truth", type=Path, required=True)
    tracking.add_argument("--predictions", type=Path, required=True)
    tracking.add_argument("--output", type=Path, required=True)
    tracking.add_argument("--iou-threshold", type=float, default=0.50)

    benchmark = subparsers.add_parser("benchmark", help="Measure pipeline runtime")
    benchmark.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    benchmark.add_argument("--source", required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--frames", type=int, default=300)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--device", help="YOLO device override")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.command == "run":
            config = load_config(args.config)
            if args.overwrite:
                config.video.overwrite = True
            if args.preserve_audio:
                config.video.preserve_audio = True
            if args.device:
                config.detector.device = args.device
            summary = run_video(
                _source(args.source),
                args.output,
                config,
                display=args.display,
                mot_output=args.mot_output,
            )
            print(json.dumps(asdict(summary), indent=2))
        elif args.command == "evaluate-detection":
            config = load_config(args.config)
            result = evaluate_detection(config, args.data, args.output, args.split)
            print(json.dumps(result, indent=2))
        elif args.command == "evaluate-tracking":
            result = evaluate_tracking(
                args.ground_truth,
                args.predictions,
                args.output,
                args.iou_threshold,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "benchmark":
            config = load_config(args.config)
            if args.device:
                config.detector.device = args.device
            result = benchmark_video(
                _source(args.source),
                config,
                args.output,
                max_frames=args.frames,
                warmup_frames=args.warmup,
            )
            print(json.dumps(result, indent=2))
        else:  # pragma: no cover - argparse enforces a command.
            parser.error(f"Unknown command: {args.command}")
    except (PerceptionEngineError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    return 0
