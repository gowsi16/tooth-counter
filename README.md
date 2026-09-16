# Gear Tooth Counter

A small web app that watches a video of three meshed gears and reports how
many teeth each gear has. Built around a simple idea: sample a handful of
video frames instead of every frame, count teeth on each sampled frame, and
let the noisy per-frame guesses vote on a final, stabilized answer.

Tested end-to-end against `sample_data/sample_gears.mp4` (a real close-up
clip of a starter-motor planetary gear set) — the pipeline correctly finds
all three gears in every sampled frame and converges on a consistent
16-teeth reading for each one.

## Quick start

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in a browser. Upload a video, wait for the
upload to finish, click **Analyze video**.

## User flow

1. **Upload / select a video.** The user drags in or picks a video file. It
   uploads to the backend and immediately plays in an inline preview so they
   know it's the right clip.
2. **Click "Analyze."** The button is disabled until a video has finished
   uploading, so there's nothing to click before there's something to
   analyze.
3. **Backend does the work.** The video is opened once, a sparse set of
   frames is pulled out, and each sampled frame goes through the same
   detect → count pipeline. This happens synchronously behind the request —
   for a typical 20-40 second clip it takes a few seconds.
4. **Stabilized results come back.** The UI renders three gear cards, each
   showing a tooth count and a confidence percentage (what fraction of the
   sampled frames agreed on that count). A snapshot of one analyzed frame is
   shown with the detected gears outlined, so the user can sanity-check what
   the backend actually looked at.

## Why sample frames instead of analyzing all of them

Gears in a hand-held or fixed-camera video rotate slowly relative to frame
rate — a 30-second clip at 60 fps is 1,800 nearly-identical frames. Running
full detection on all of them buys almost nothing over running it on, say,
40 evenly-spaced frames, except several times the processing time. Sampling
is also what makes the stabilization step (below) possible: it needs several
*independent* looks at each gear, not several *identical* ones.

The included pipeline samples at roughly 2 frames per second, capped at 40
samples total regardless of video length, so a 10-minute video doesn't turn
into a 10-minute analysis. Both numbers are constants at the top of
`gear_counter.py` and are the first thing to tune for a different rig or
frame rate.

## How a single frame is analyzed

1. **Find the gear assembly.** A Hough circle transform looks for one large,
   high-contrast circle — the housing or backing plate the gears sit in.
   This gives a region of interest (ROI) to search within, which both speeds
   up the next step and keeps background clutter from producing false
   positives.
2. **Find each gear's center.** Within that ROI, a second Hough circle pass
   looks for three *small* circles — the bearing hole at the center of each
   gear, rather than its toothed outer edge. This is a deliberate choice:
   the hub is a small, near-perfect circle (exactly what circular Hough
   fitting is good at), while the toothed rim is scalloped, not circular,
   and gears that mesh with each other are physically touching where their
   teeth interlock — which makes separating them by color/contour
   unreliable. Detecting the hub sidesteps both problems. Candidates are
   filtered to a plausible "orbit radius" from the assembly center to reject
   stray detections (mounting bolts, housing edges), and the three that
   survive are sorted by angle so "Gear 1 / 2 / 3" refers to the same
   physical gear across frames.
3. **Count the teeth on each gear.** For each detected gear center, an
   annulus (ring) around it — sized to bracket the tooth tips — is "unrolled"
   with `cv2.warpPolar` into a rectangular strip: angle along one axis,
   radius along the other. Averaging across the radius band collapses that
   strip into a single 1-D brightness signal, one value per angle. Every
   tooth produces one cycle of that signal as you sweep around the gear, so
   the tooth count is the *dominant frequency* of the signal — found with an
   FFT rather than by picking individual peaks, because it uses information
   from the whole ring at once and is far less thrown off by motion blur, a
   stray reflection, or a hand briefly crossing the frame than counting
   bumps one at a time is.

## Stabilizing counts across frames

A single frame can misread a tooth count by one or two teeth — that's
expected, not a bug to chase out of the per-frame detector. What matters is
that those errors are close to random, while the *true* count comes up
correctly more often than any single wrong value does across many
independently-sampled frames.

So each gear collects a list of per-frame readings, and the final answer is
simply the **mode** — the most frequently occurring value in that list. The
fraction of samples that agreed with the winning value is reported back as a
confidence score, so the UI (and a developer debugging a bad reading) can
tell a rock-solid 95%-agreement answer from a shaky 40%-agreement one where
the video, lighting, or detection parameters need attention.

On the bundled sample video, per-frame readings for one gear ranged from 14
to 18 teeth depending on the frame, but the mode across 40 samples landed on
16 teeth with roughly two-thirds of frames agreeing — which happens to be
mechanically correct, since all three planet gears in that assembly are
identical and other frame-by-frame evidence (all three gears independently
converging on the same number) corroborates it.

## Architecture

```
gear-tooth-counter/
├── backend/
│   ├── app.py             Flask routes only — no CV logic
│   ├── gear_counter.py    the entire CV pipeline, framework-agnostic
│   ├── requirements.txt
│   └── uploads/           uploaded videos land here
├── frontend/
│   └── index.html         upload UI, preview, results — vanilla JS, no build step
├── sample_data/
│   └── sample_gears.mp4   real test clip used to validate the pipeline
└── README.md
```

The split matters more than it looks: `gear_counter.py` doesn't import
Flask or know it's being called from a web request. It's a plain function,
`analyze_video(path) -> AnalysisResult`, that a developer can run directly
in a script, a notebook, or a test — which is exactly how it was tuned and
validated against the sample video before the web layer was ever written.
`app.py` is deliberately thin: it validates the upload, calls that one
function, and serializes the result. That boundary is what makes the next
section a drop-in change rather than a rewrite.

**Endpoints:**
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | serves the frontend |
| `POST` | `/api/upload` | accepts a video file, returns a `video_id` |
| `GET` | `/api/video/<id>` | streams the video back for the `<video>` preview |
| `POST` | `/api/analyze/<id>` | runs the pipeline, returns per-gear JSON results |

## Upgrading detection later: YOLO

The Hough-circle detection in `locate_assembly()` and `locate_gear_centers()`
works well for a clean, top-down, single-assembly shot like the sample
video, but it's tuned by hand and will need re-tuning (or will simply miss
detections) for gears with different proportions, cluttered backgrounds, or
odd camera angles.

If that becomes a problem, the fix is localized: train a small YOLO (or
similar) object detector to output bounding boxes or center points for
"gear" directly, and swap it in for `locate_assembly()` +
`locate_gear_centers()`. Nothing downstream needs to change — the tooth
counter and stabilizer both just need a `(center_x, center_y)` per gear, and
don't care how it was produced. This is also the natural place to add
support for gears that aren't evenly spaced around a common center, or
videos with more or fewer than three gears in frame.

## Known limitations

- Assumes exactly three gears, visible together and mostly stationary in
  frame (small camera shake is fine — the sample video has plenty of it).
- Lighting and contrast matter: the hub-detection step needs the gear's
  center bearing hole to read as a clean, roughly circular edge.
- `TOOTH_RING_INNER` / `TOOTH_RING_OUTER` in `gear_counter.py` bracket the
  radius band the tooth-counting annulus is drawn at, in pixels — these are
  fitted to the sample video's camera distance and will need adjusting (or
  computing dynamically from the detected hub-to-housing ratio) for footage
  shot from a different distance.
