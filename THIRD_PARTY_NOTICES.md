# Third-party notices

This project depends on third-party software. Dependency versions are constrained in `pyproject.toml`; every installation must also comply with the dependency's own license.

| Component | Purpose | License / terms | Project |
|---|---|---|---|
| Ultralytics | YOLO11 model loading, inference, and detection validation | AGPL-3.0 or Ultralytics Enterprise license | <https://github.com/ultralytics/ultralytics> |
| PyTorch / TorchVision | Model execution | BSD-style and bundled third-party notices | <https://github.com/pytorch/pytorch> |
| OpenCV Python | Video decoding, encoding, and drawing | Apache-2.0 and bundled third-party notices | <https://github.com/opencv/opencv-python> |
| NumPy | Array operations | BSD-3-Clause | <https://github.com/numpy/numpy> |
| SciPy | Hungarian optimization | BSD-3-Clause | <https://github.com/scipy/scipy> |
| PyYAML | Configuration loading | MIT | <https://github.com/yaml/pyyaml> |
| motmetrics | MOT metric accumulation and summaries | MIT | <https://github.com/cheind/py-motmetrics> |

The project does not redistribute YOLO model weights. Ultralytics downloads the configured model to the user's local cache when needed. Review the current Ultralytics license and model terms before deployment, redistribution, or commercial use.

Development-only tools—including pytest, Ruff, mypy, and coverage.py—are not incorporated into runtime output.
