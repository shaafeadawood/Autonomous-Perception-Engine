from __future__ import annotations

from pathlib import Path

from autonomous_perception_engine.cli import build_parser, main


def test_run_command_parses_expected_arguments() -> None:
    args = build_parser().parse_args(
        ["run", "--source", "input.mp4", "--output", "output.mp4", "--device", "cpu"]
    )
    assert args.command == "run"
    assert args.source == "input.mp4"
    assert args.device == "cpu"


def test_missing_configuration_returns_user_error(tmp_path: Path) -> None:
    result = main(
        [
            "run",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--source",
            "input.mp4",
            "--output",
            "output.mp4",
        ]
    )
    assert result == 2
