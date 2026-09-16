"""
gear_counter.py
================
Core computer-vision pipeline for the Gear Tooth Counter app.

This module has one job: given a path to a video file, figure out how many
teeth each of the (up to three) gears in the frame has, and how confident we
are in that answer.

The pipeline, in order:
  1. sample_frames()          -> pull out a handful of frames instead of every frame
  2. locate_assembly()        -> find the round gear housing in a frame (ROI)
  3. locate_gear_centers()    -> find the 3 gear hub centers inside that ROI
  4. count_teeth_in_frame()   -> "unroll" each gear into a 1-D signal and count
                                  the repeating tooth pattern in it
  5. stabilize_counts()       -> combine the per-frame guesses into one
                                  confident final number per gear

Everything here is deliberately classical OpenCV (Hough circles, polar
unwarping, FFT) rather than a trained model. See the README for why, and for
where a YOLO-based detector would slot in later without changing steps 4-5.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Tunable constants. These were fitted against real starter-motor planetary
# gear footage (see sample_data/sample_gears.mp4) and are exposed here, in one
# place, so a developer adapting this to a different gear rig only has to
# touch this block, not hunt through the pipeline functions below.
# --------------------------------------------------------------------------

NUM_GEARS = 3                     # how many gears we expect to find
SAMPLE_FPS = 2                    # analyze ~2 frames per second of video
MAX_SAMPLES = 40                  # hard cap, so a 10-minute video doesn't crawl

OUTER_RING_MIN_R = 100            # px, radius search range for the gear
OUTER_RING_MAX_R = 260            # housing / assembly (tune per camera distance)

HUB_MIN_R = 10                    # px, radius search range for each gear's
HUB_MAX_R = 30                    # center bearing hole

HUB_BAND_MIN = 30                 # a detected hub is only accepted if its
HUB_BAND_MAX = 140                # distance from the ROI center falls in
                                   # this band (filters out false positives
                                   # like mounting bolts / housing edges)

TOOTH_RING_INNER = 30             # px, inner radius of the annulus we unwrap
TOOTH_RING_OUTER = 60             # px, outer radius of the annulus we unwrap
                                   # (should bracket the tooth tips)

TEETH_COUNT_MIN = 8               # sane bounds on "how many teeth could a
TEETH_COUNT_MAX = 40              # gear plausibly have" - keeps the frequency
                                   # search from latching onto noise


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class GearResult:
    gear_id: int                       # 1, 2, 3 - stable across the video
    label: str                         # "Gear 1" etc, for the UI
    final_count: Optional[int]         # the stabilized answer (None if never seen)
    confidence: float                  # fraction of samples that agreed (0-1)
    samples_used: int                  # how many frames contributed a reading
    per_frame_counts: List[int] = field(default_factory=list)  # raw readings


@dataclass
class AnalysisResult:
    gears: List[GearResult]
    frames_sampled: int
    frames_with_detection: int
    annotated_frame_b64: Optional[str]  # one labeled frame, for the UI to show
    annotated_timestamp_sec: Optional[float] = None # timestamp of the annotated frame


# --------------------------------------------------------------------------
# Step 1: frame sampling
# --------------------------------------------------------------------------

def sample_frames(video_path: str, target_fps: float = SAMPLE_FPS,
                   max_samples: int = MAX_SAMPLES,
                   start_time_sec: float = 0.0) -> List[Tuple[float, np.ndarray]]:
    """
    Pull a sparse set of frames out of the video instead of decoding and
    analyzing every single one.

    Why this matters: a 30-second clip at 60 fps is 1,800 frames. Gear
    rotation is slow relative to frame rate, so consecutive frames are
    almost identical - running full detection on every one of them is
    almost pure waste. Sampling at ~2 frames/sec gives the stabilization
    step (step 5) plenty of independent looks at the gears while keeping
    the analysis a few seconds instead of a few minutes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / target_fps)))

    # If sampling at `step` would give us more than max_samples, widen the
    # step so we still cover the whole video rather than just the start.
    start_frame = int(start_time_sec * src_fps)
    remaining_frames = max(0, total_frames - start_frame)
    expected = remaining_frames // step if step else remaining_frames
    if expected > max_samples and max_samples > 0:
        step = max(1, remaining_frames // max_samples)

    frames = []
    idx = start_frame
    while len(frames) < max_samples and idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frames.append((timestamp, frame))
        idx += step
        if idx >= total_frames:
            break

    cap.release()
    return frames


# --------------------------------------------------------------------------
# Step 2: locate the round gear assembly (housing) in a frame
# --------------------------------------------------------------------------

def locate_assembly(gray_blurred: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Find the large circular housing that holds the gears, via a Hough
    circle transform tuned for a big, high-contrast circle. Returns
    (x, y, r) in full-frame coordinates, or None if nothing was found.

    This gives us a region of interest (ROI) to search within for the
    individual gears, which both speeds up the next step and avoids false
    positives from background clutter outside the assembly.
    """
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=1000,
        param1=100, param2=60,
        minRadius=OUTER_RING_MIN_R, maxRadius=OUTER_RING_MAX_R,
    )
    if circles is None:
        return None
    x, y, r = np.round(circles[0, 0]).astype(int)
    return int(x), int(y), int(r)


# --------------------------------------------------------------------------
# Step 3: locate the individual gear centers inside the ROI
# --------------------------------------------------------------------------

def locate_gear_centers(roi_gray: np.ndarray, roi_center: Tuple[float, float],
                         num_gears: int = NUM_GEARS) -> List[Tuple[int, int, int]]:
    """
    Find each gear's center within the cropped ROI.

    We deliberately target the gear's *hub bearing hole* rather than its
    toothed outer edge. The hub is a small, near-perfect, high-contrast
    circle, which is exactly what a circular Hough transform is good at
    fitting. The toothed rim, by contrast, is scalloped rather than
    circular, and gears that mesh with each other are physically touching
    at the tooth line - which makes contour/blob-based separation
    unreliable. Detecting the hub sidesteps both problems.

    Candidates are filtered to a plausible "orbit radius" band around the
    ROI center (real gear hubs cluster at a consistent distance from the
    assembly center; stray detections like a mounting bolt hole usually
    don't), then the closest `num_gears` candidates to that band are kept
    and sorted by angle so gear identity stays consistent frame to frame.
    """
    hubs = cv2.HoughCircles(
        roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
        param1=80, param2=22, minRadius=HUB_MIN_R, maxRadius=HUB_MAX_R,
    )
    if hubs is None:
        return []

    cx, cy = roi_center
    candidates = []
    for hx, hy, hr in np.round(hubs[0]).astype(int):
        dist = float(np.hypot(hx - cx, hy - cy))
        if HUB_BAND_MIN <= dist <= HUB_BAND_MAX:
            candidates.append((int(hx), int(hy), int(hr), dist))

    if len(candidates) < num_gears:
        return []

    # Prefer candidates whose orbit radius agrees with the group's median -
    # this is what pushes out one-off false positives.
    median_dist = float(np.median([c[3] for c in candidates]))
    candidates.sort(key=lambda c: abs(c[3] - median_dist))
    chosen = candidates[:num_gears]

    # Sort by angle around the group's own centroid so "Gear 1/2/3" refers
    # to a consistent physical gear across frames (as long as the camera
    # and housing don't rotate wildly between samples).
    gx = np.mean([c[0] for c in chosen])
    gy = np.mean([c[1] for c in chosen])
    chosen.sort(key=lambda c: np.arctan2(c[1] - gy, c[0] - gx))

    return [(x, y, r) for x, y, r, _ in chosen]


# --------------------------------------------------------------------------
# Step 4: count the teeth on one gear in one frame
# --------------------------------------------------------------------------

def count_teeth_in_frame(gray_img: np.ndarray, center: Tuple[int, int],
                          r_inner: int = TOOTH_RING_INNER,
                          r_outer: int = TOOTH_RING_OUTER,
                          angle_bins: int = 1440,
                          force_count: Optional[int] = None) -> Tuple[Optional[int], List[float]]:
    """
    Count the teeth on a single gear by "unrolling" the ring of teeth into
    a 1-D signal and measuring how many times a repeating pattern occurs.

    How it works:
      1. cv2.warpPolar turns the circular tooth ring into a rectangular
         strip: angle along one axis, radius along the other. A tooth
         sticking out now shows up as a bright/dark bump at a particular
         angle, rather than a hard-to-count shape on a circle.
      2. Average across the radius band to collapse the strip into a
         single 1-D brightness profile, one value per angle.
      3. Each tooth produces one cycle of that profile as we sweep all the
         way around the gear, so the *tooth count is just the dominant
         frequency of this signal*. We get that with an FFT and pick the
         strongest frequency bin within a plausible tooth-count range.

    We use the FFT's dominant frequency rather than naive peak-picking on
    the profile because it averages information from the whole ring at
    once, which turned out to be far less sensitive to motion blur,
    uneven lighting, and partial occlusion (e.g. a hand or wire crossing
    the frame) than trying to count individual bumps directly - a single
    misdetected bump throws off a peak-counter's total, but has much less
    effect on which frequency dominates the whole signal.
    """
    cx, cy = float(center[0]), float(center[1])
    if r_outer <= r_inner:
        return None, []

    polar = cv2.warpPolar(
        gray_img, (r_outer, angle_bins), (cx, cy), r_outer, cv2.WARP_POLAR_LINEAR
    )
    band = polar[:, r_inner:r_outer].astype(np.float32)
    if band.size == 0:
        return None, []

    profile = band.mean(axis=1)
    profile = profile - profile.mean()

    window = np.hanning(len(profile))
    fft_result = np.fft.rfft(profile * window)
    spectrum = np.abs(fft_result)

    if force_count is not None:
        dominant = force_count
    else:
        lo, hi = TEETH_COUNT_MIN, min(TEETH_COUNT_MAX, len(spectrum) - 1)
        if hi <= lo:
            return None, []
        dominant = int(np.argmax(spectrum[lo:hi]) + lo)
    
    # Reconstruct the dominant wave to find tooth angles
    dominant_wave_fft = np.zeros_like(fft_result)
    dominant_wave_fft[dominant] = fft_result[dominant]
    smoothed_profile = np.fft.irfft(dominant_wave_fft, n=len(profile))
    
    peak_angles = []
    for i in range(len(smoothed_profile)):
        prev_i = (i - 1) % len(smoothed_profile)
        next_i = (i + 1) % len(smoothed_profile)
        if smoothed_profile[i] > smoothed_profile[prev_i] and smoothed_profile[i] > smoothed_profile[next_i]:
            angle = i * 2 * np.pi / angle_bins
            peak_angles.append(angle)
            
    return dominant, peak_angles


# --------------------------------------------------------------------------
# Step 5: combine per-frame guesses into one stabilized answer
# --------------------------------------------------------------------------

def stabilize_counts(per_frame_counts: List[int]) -> Tuple[Optional[int], float]:
    """
    Turn a list of noisy per-frame tooth counts into one confident answer.

    Individual frames can misread the count by a tooth or two - motion
    blur, an awkward viewing angle, a hand passing in front of the gear,
    etc. But those errors are close to random, while the *true* count
    shows up correctly far more often than any one wrong value. Taking
    the mode (most frequent reading) across many independently-sampled
    frames cancels the noise out, which is the whole reason we sample
    many frames instead of just trusting a single snapshot.

    Returns (final_count, confidence) where confidence is the fraction of
    samples that agreed with the winning count - a simple, honest signal
    for the UI to show the user (and a hook for "please retry with a
    clearer video" if it comes out too low).
    """
    if not per_frame_counts:
        return None, 0.0
    counts = Counter(per_frame_counts)
    best_value, best_freq = counts.most_common(1)[0]
    confidence = best_freq / len(per_frame_counts)
    return best_value, confidence


# --------------------------------------------------------------------------
# Orchestration: tie steps 1-5 together for a full video
# --------------------------------------------------------------------------

def _draw_annotations(frame: np.ndarray, roi_origin: Tuple[int, int],
                       gears: List[Tuple[int, int, int]],
                       labels: List[str],
                       gear_peaks: List[List[float]]) -> np.ndarray:
    """Draw the detected gear circles + labels on a copy of the frame, for
    display in the UI so the user can see what the backend "saw"."""
    out = frame.copy()
    ox, oy = roi_origin
    for (x, y, r), label, peaks in zip(gears, labels, gear_peaks):
        center = (x + ox, y + oy)
        cv2.circle(out, center, TOOTH_RING_INNER, (0, 200, 0), 1)
        cv2.circle(out, center, TOOTH_RING_OUTER, (0, 200, 0), 1)
        cv2.circle(out, center, 3, (0, 0, 255), -1)
        
        for idx, angle in enumerate(peaks):
            px1 = int(center[0] + TOOTH_RING_INNER * np.cos(angle))
            py1 = int(center[1] + TOOTH_RING_INNER * np.sin(angle))
            px2 = int(center[0] + TOOTH_RING_OUTER * np.cos(angle))
            py2 = int(center[1] + TOOTH_RING_OUTER * np.sin(angle))
            cv2.line(out, (px1, py1), (px2, py2), (0, 255, 255), 2)
            
            text_r = (TOOTH_RING_INNER + TOOTH_RING_OUTER) // 2
            tx = int(center[0] + text_r * np.cos(angle))
            ty = int(center[1] + text_r * np.sin(angle))
            text = str(idx + 1)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            # draw a thick black outline for legibility over the yellow lines
            cv2.putText(out, text, (tx - w // 2, ty + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (tx - w // 2, ty + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw the label a bit higher so it doesn't overlap with the top numbers
        cv2.putText(out, label, (center[0] - 25, center[1] - TOOTH_RING_OUTER - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
    return out


def analyze_video(video_path: str, num_gears: int = NUM_GEARS,
                   target_fps: float = SAMPLE_FPS,
                   start_time_sec: float = 0.0) -> AnalysisResult:
    """
    Full pipeline entry point: sample frames from the video, detect and
    count teeth per gear per frame, then stabilize into a final answer.
    This is the one function the Flask route needs to call.
    """
    frames = sample_frames(video_path, target_fps=target_fps, start_time_sec=start_time_sec)
    per_gear_counts: List[List[int]] = [[] for _ in range(num_gears)]
    frames_with_detection = 0
    annotated_frame_info = None

    for timestamp, frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)

        outer = locate_assembly(gray_blur)
        if outer is None:
            continue
        ox, oy, orad = outer
        pad = 5
        x0, y0 = max(0, ox - orad - pad), max(0, oy - orad - pad)
        x1, y1 = ox + orad + pad, oy + orad + pad
        roi = gray_blur[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        rh, rw = roi.shape
        roi_center = (rw / 2.0, rh / 2.0)

        gear_centers = locate_gear_centers(roi, roi_center, num_gears=num_gears)
        if len(gear_centers) < num_gears:
            continue

        frames_with_detection += 1
        frame_counts = []
        for gx, gy, _gr in gear_centers:
            n, _ = count_teeth_in_frame(roi, (gx, gy))
            frame_counts.append(n)

        for i, n in enumerate(frame_counts):
            if n is not None:
                per_gear_counts[i].append(n)

        # Save the first clean detection frame so we can annotate it later
        if annotated_frame_info is None:
            annotated_frame_info = (frame.copy(), roi.copy(), x0, y0, gear_centers, timestamp)

    gears = []
    final_counts = []
    for i in range(num_gears):
        final_count, confidence = stabilize_counts(per_gear_counts[i])
        final_counts.append(final_count)
        gears.append(GearResult(
            gear_id=i + 1,
            label=f"Gear {i+1}",
            final_count=final_count,
            confidence=round(confidence, 2),
            samples_used=len(per_gear_counts[i]),
            per_frame_counts=per_gear_counts[i],
        ))
        
    annotated_b64 = None
    annotated_ts = None
    if annotated_frame_info is not None:
        frame, roi, x0, y0, gear_centers, ts = annotated_frame_info
        frame_peaks = []
        for i, (gx, gy, _gr) in enumerate(gear_centers):
            # force the FFT to find angles specifically for the stabilized final count
            _, peaks = count_teeth_in_frame(roi, (gx, gy), force_count=final_counts[i])
            frame_peaks.append(peaks)
            
        labels = [f"Gear {i+1}" for i in range(num_gears)]
        annotated = _draw_annotations(frame[y0:y1, x0:x1], (0, 0),
                                       gear_centers, labels, frame_peaks)
        ok, buf = cv2.imencode(".jpg", annotated)
        if ok:
            annotated_b64 = base64.b64encode(buf).decode("ascii")
            annotated_ts = ts

    return AnalysisResult(
        gears=gears,
        frames_sampled=len(frames),
        frames_with_detection=frames_with_detection,
        annotated_frame_b64=annotated_b64,
        annotated_timestamp_sec=annotated_ts,
    )
