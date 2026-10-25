# Evaluation methodology

Runtime demonstrations and accuracy evaluation answer different questions. This project records them separately and does not derive accuracy from an unlabeled video.

## Detection evaluation

Detection evaluation requires a YOLO-format dataset YAML with labeled validation images. Run:

```bash
ape evaluate-detection \
  --config configs/default.yaml \
  --data path/to/dataset.yaml \
  --output outputs/detection-metrics.json
```

The command delegates matching and AP calculation to the installed Ultralytics validator and records:

- mean precision and recall;
- mAP@0.50;
- mAP@0.50:0.95;
- mAP@0.75 when exposed by the installed version;
- available per-class mAP@0.50:0.95;
- model, split, dataset path, timestamp, and configuration.

Results are meaningful only for the named dataset, split, annotation policy, model artifact, and evaluation settings. Do not compare figures from different protocols as though they were interchangeable.

## Tracking evaluation

Tracking evaluation requires ground-truth identities, not detection boxes alone. Generate predictions during a normal run:

```bash
ape run \
  --source path/to/sequence.mp4 \
  --output outputs/sequence.mp4 \
  --mot-output outputs/sequence.txt
```

Then compare MOTChallenge-format files:

```bash
ape evaluate-tracking \
  --ground-truth path/to/gt.txt \
  --predictions outputs/sequence.txt \
  --output outputs/tracking-metrics.json \
  --iou-threshold 0.5
```

Reported metrics include IDF1, MOTA, MOTP, identity switches, misses, false positives, fragmentation, precision, and recall. In `motmetrics`, MOTP is the average assignment distance and lower is better. For HOTA, use the same exported predictions with the TrackEval reference implementation and document its exact configuration.

## Runtime benchmarking

```bash
ape benchmark \
  --source path/to/sequence.mp4 \
  --output outputs/runtime.json \
  --frames 300 \
  --warmup 10
```

The benchmark:

- loads the configured real model;
- synchronizes pending CUDA work before ending detector timing;
- excludes a configurable warm-up prefix;
- separates read, detection, tracking, and total latency;
- reports mean, median, p95, minimum, and maximum time;
- derives FPS from mean end-to-end measured latency;
- records environment and configuration data.

Rendering and video encoding are excluded from the benchmark command by design. Full-run manifests include those stages when measuring deployed file-processing behavior.

## Current verified status

The repository's deterministic unit and synthetic integration tests validate software behavior. No road-dataset accuracy results are currently published because no appropriately labeled evaluation dataset has been selected and run. No hardware FPS claim is published because performance depends on the model, device, resolution, codec path, and source.

When results are added, commit the JSON outputs or a compact derived table alongside:

- dataset name, version, license, and split;
- model filename and checksum;
- configuration file and commit hash;
- hardware and package versions;
- the untouched machine-readable result.
