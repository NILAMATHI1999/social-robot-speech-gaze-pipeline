# Lighting test: L2CS gaze input in the dark vs. phone torches (21 Oct 2026)

## Setup
- Laptop webcam (HP True Vision, 640x480), RetinaFace + L2CS ResNet50 (Gaze360), CPU.
- Same face-quality rule as `step1d_l2cs_gaze.py`: face width >= 60 px, face score >= 0.50,
  face brightness >= 20 (mean grey value of the face box, 0-255).
- RetinaFace confidence lowered to 0.3 for this test so that weak detections are visible.
- One person, seated, looking at the camera. About 27 s, 116 frames.
- Raw data: `lighting_test_20261021_raw.csv`. The condition labels in the CSV were not
  recorded (all rows say `no_light`); the phases below were identified from the brightness jumps.

## Results

| Phase | Time | Room brightness | Face brightness | Face score | Usable for gaze | Gaze yaw / pitch |
|---|---|---|---|---|---|---|
| Dark | 0-6.5 s | 1.2-1.4 | 1.4-2.1 | 0.31-0.99 | 0 % (`too_dark`, `no_face`, `low_score`) | about -75 / -56, meaningless |
| Light being switched on / adjusted | 6.7-11 s | 10-23 | 31-106 | 0.99-1.00 | 100 % | yaw jumps from -102 to +70, unstable |
| Two phone torches in front of the face | 11.1-22.8 s | 23-42 | 126-154 | 0.99-1.00 | 100 % | yaw -12 to +4, pitch -8 to -21, stable |
| Dark again | 23-26.7 s | 1.2-3.8 | 1.3-2.4 | 0.48-1.00 | 0 % | about -70 / -58, meaningless |

Frame rate on the laptop CPU: median 4.6 FPS.

## Findings
1. **The brightness rule works.** Dark frames had a face brightness of at most 2.4; lit frames had at least 31.
   The threshold of 20 separates them cleanly in this test.
2. **The face score does not detect darkness.** RetinaFace reported scores of 0.90-0.99 on frames with a
   face brightness of about 2. The 0.9 face-score threshold of Step 2B therefore does not protect
   against darkness; the brightness check is needed in addition.
3. **In the dark, L2CS returns a constant, plausible-looking wrong answer** (yaw about -75 deg,
   pitch about -57 deg), which would be classified as "not looking". Without the brightness check
   this would wrongly lower the attention score. With the check it becomes `unknown`, as intended.
4. **With two phone torches the gaze is stable** (yaw within about 15 deg, pitch -8 to -21 deg while
   looking at the camera) and every frame is usable.
5. **Room brightness is a poor indicator.** With the torches the whole image only reached 23-42,
   while the face reached 126-154. The check must use the face region, as it already does.

## Limitations of the method
- Gaze is not available at all without light on the face: 0 % usable frames in the dark. The
  partner selection then depends on head/body rules (and YOLO pose may also degrade in the dark;
  not tested here).
- A strong frontal light source is needed. Normal room light was not tested in this run.
- During the switching phase (face brightness 31-106) the gaze was unstable although the frames
  passed the quality rule. The cause (movement while adjusting the torches vs. uneven light) cannot
  be separated without labels; this should be repeated with a fixed light level.
- Single run, one person, one camera, about 27 s. Repeat with labelled phases, normal room light,
  side light and more than one person before drawing final conclusions.
- CPU speed (about 4-5 FPS) limits the number of gaze samples; the Jetson GPU will be faster.
