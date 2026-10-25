# Demonstration video workflow

## Media policy

Do not commit or publish footage merely because it is publicly viewable. Confirm that the license or written permission permits redistribution and creation of an annotated derivative.

Tesla footage and derived outputs are intentionally excluded while their redistribution rights remain unverified. If suitable permission is obtained, the intended output filename is:

```text
Tesla Autonomous Driving - Real-Time Object Detection and Tracking.mp4
```

Use that name only if Tesla footage is later authorized. Otherwise select a title that accurately identifies the replacement footage.

## Recommended sources

Suitable options include:

- footage recorded by the project owner;
- a dataset sample whose license explicitly permits redistribution and derivatives;
- a stock or Creative Commons clip whose attribution and commercial-use conditions fit the repository;
- a short synthetic road scene used solely to demonstrate software mechanics.

Record the source URL, author, exact license, retrieval date, and required attribution before processing.

## Generate the demonstration

Place the source under ignored local data:

```text
data/input/licensed-road-video.mp4
```

Run:

```powershell
ape run `
  --config configs/default.yaml `
  --source "data/input/licensed-road-video.mp4" `
  --output "outputs/Autonomous-Perception-Engine-Demo.mp4" `
  --mot-output "outputs/demo-tracks.txt" `
  --preserve-audio
```

Review the entire output for false positives, lost tracks, ID switches, unreadable overlays, and accidental personal information. The generated run manifest should be retained with the demo record.

## Publish without bloating Git

Prefer a GitHub Release asset or an external host that preserves the media license. Add a poster image or short optimized preview under a clearly documented license, then link the full video prominently from the README.

Before publication verify:

- redistribution and derivative rights;
- attribution text and links;
- absence of sensitive location, face, plate, or account information where applicable;
- output size and browser-compatible encoding;
- correspondence between README claims, video configuration, and run manifest.
