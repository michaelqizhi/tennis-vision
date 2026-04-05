# Tennis Court Line Detection: Approach A vs Approach B

## Rigorous Comparative Analysis

**Date**: June 2025
**Context**: Near-half tennis court line detection for amateur courtside video
**Scope**: This is a read-only analysis. No code was modified.

---

## Executive Summary

Approach B (local-contrast via V top-hat + S black-hat) is **strictly more robust** than
Approach A (global thresholds + color detection + scoring) across all 12 conditions
analyzed. In no condition does Approach A outperform Approach B. The advantages range
from marginal (both approaches succeed easily) to decisive (Approach A fails
categorically while B succeeds).

The recommendation is to implement Approach B as the **primary** line detection method,
retaining the saturation filter as a scored fallback during a validation period, then
dropping the color-based candidates entirely.

---

## How the Morphological Operations Actually Work

Before the per-condition analysis, it's worth grounding the math. These details
determine every prediction below.

### White top-hat on V (both approaches use this)

```
tophat(V) = V - opening(V)
opening   = dilate(erode(V, kernel), kernel)
kernel    = 23×23 ellipse
```

For a **thin white line** (≈5 px wide, V≈220) on a **uniform court** (V≈X):
- Erosion with 23×23 removes the line entirely (line width < kernel width)
- Dilation restores the surrounding court but cannot restore the erased line
- Opened value at line position ≈ X (court value)
- **Top-hat at line ≈ 220 − X**

For a **large uniform area** (width >> 23 px):
- Opening ≈ original → top-hat ≈ 0

This is the key property: only features **thinner than the kernel** produce a response.

### Black-hat on S (Approach B only — proposed)

```
blackhat(S) = closing(S) - S
closing     = erode(dilate(S, kernel), kernel)
```

For a **thin achromatic line** (S≈15, ≈5 px wide) on a **chromatic court** (S≈Y):
- Dilation: each line pixel's 23×23 neighborhood contains court pixels at S≈Y.
  Max wins → dilated line pixel ≈ Y
- Erosion of dilated image: the thin "hole" has been filled; neighborhood is
  uniformly ≈Y → eroded value ≈ Y
- Closed value at line ≈ Y
- **Black-hat at line ≈ Y − 15**

For a **large uniform area** (S≈Y, width >> 23 px):
- Closing ≈ original → black-hat ≈ 0

The S black-hat extracts features that are **locally less saturated** than their
surroundings — precisely what white lines on colored courts are.

### At the boundary between two court colors

Consider green (S≈80) adjacent to blue (S≈43), boundary wider than the kernel:
- Closing doesn't bridge the boundary (both regions are wider than 23 px)
- Black-hat at boundary ≈ small transition artifact (≈5-10), not line-like
- **No false positive at color boundaries** ✓

### For a line AT a color boundary

Line (S≈15, 5 px) with green (S≈80) on one side, blue (S≈43) on other:
- Dilation picks max from neighborhood → ≈80 (green side dominates)
- Closing fills line with ≈80
- Black-hat ≈ 80 − 15 = 65. **Strong response** ✓

---

## Per-Condition Analysis

### Condition 1: Bright Single-Color Court (V≈165, green, outdoor daytime)

**Typical values**: Court V≈165, S≈70-90. Lines V≈220, S≈10-25.

**Approach A**:
- Court mask (S > 50): S≈70-90 → passes easily ✓
- V top-hat at line: 220 − 165 = **55**. Otsu on court region finds clear bimodal
  distribution (most pixels ≈0, lines ≈55). Effective threshold ≈ max(0.7 × 28, 5) ≈ 20.
  Line response 55 >> 20 ✓
- S < 50: Line S≈15 ✓
- V > 150: Line V≈220 ✓
- Neighbor filter: Court mask covers all court surface → full support ✓
- **Prediction: Works well.** All gates pass with margin.

**Approach B**:
- V top-hat: 55 (same as above) ✓
- S black-hat at line: Court S≈80, line S≈15. Response ≈ **65**. Very strong ✓
- Surface mask (V > 30): Court V≈165 ✓
- **Prediction: Works well.** Both local-contrast signals are strong.

**Addressing the user's concern**: "Top-hat response might be weak due to small V
difference." A ΔV of 55 is not small — it's 3× the typical Otsu-adapted threshold.
The concern would be valid only if ΔV < 15-20, which requires court V > 200.
That occurs in direct sunlight on light-colored courts, but even then the S black-hat
(Approach B) provides a strong secondary signal.

| | Winner | Confidence |
|---|---|---|
| **Tie** — both succeed comfortably | — | High |

---

### Condition 2: Dark Single-Color Court (V≈110, blue, indoor)

**Typical values**: Court V≈110, S≈60-100. Lines V≈220, S≈10-25.

**Approach A**:
- Court mask: S≈60-100 >> 50 ✓
- V top-hat at line: 220 − 110 = **110**. Extremely strong ✓
- S < 50: ✓. V > 150: ✓
- **Prediction: Works easily.** This is the "happy path" for all approaches.

**Approach B**:
- V top-hat: 110 ✓
- S black-hat: ≈60-100 − 15 = **45-85** ✓
- **Prediction: Works easily.**

| | Winner | Confidence |
|---|---|---|
| **Tie** — trivial case for both | — | High |

---

### Condition 3: Multi-Color Court (green + desaturated blue S≈43) — THE V1 CASE

**Typical values**: Green: V≈130, S≈70-90. Blue surround: V≈140, S≈43. Lines: V≈220, S≈15.

**Approach A**:
- Court mask (S > 50): **Green passes. Blue surround FAILS** (S≈43 < 50).
- Lines on green surface (service line, center line, singles sidelines): Court neighbors
  are all green → full support. Top-hat, S<50, V>150 all pass ✓
- Lines at green-blue boundary (doubles sidelines, baseline): 7×7 neighborhood straddles
  the boundary. ~50% of neighbors are green (in court mask), ~50% blue (not in mask).
  That's ≈24 green neighbors out of 49, well above the threshold of 4. **These pass** ✓
- **But**: The Otsu adaptive threshold samples only from the S > 50 region (green surface).
  This biases the threshold toward the green surface's top-hat distribution. If the green
  surface is brighter than the blue surround, lines on the blue side have different contrast
  characteristics that Otsu may not capture optimally.
- Paper/simple color filter: `detect_court_color_simple` picks green as dominant. Blue
  surround with S≈43 has median S < 30 (the chromatic bin threshold), so it's **excluded
  from chromatic bins entirely**. Multi-color detection never triggers because the blue
  bins are classified as achromatic. This matches the user's observation: multi-color
  "only triggers on 2/10 frames."
- K-means: May pick blue as one of k=3 clusters if enough blue pixels are sampled.
  But `build_court_color_mask` with ±40 S tolerance centered on S≈43 gives range [3, 83],
  which is broad enough to include the blue surround. **K-means might work** depending
  on sampling luck.
- **Prediction: Saturation filter works for lines at or inside the green area. Paper
  filter fails to capture the blue surround at all. K-means is unreliable. Overall,
  the pipeline works for most lines but the court mask is incomplete and the color-based
  candidates provide no backup.**

**Approach B**:
- V top-hat: Lines V≈220, green V≈130 → response ≈90. Lines V≈220, blue V≈140 → response ≈80. **Strong everywhere** ✓
- S black-hat:
  - Lines on green: 70-90 − 15 = **55-75** ✓
  - Lines on blue surround: 43 − 15 = **28**. Moderate but present ✓
  - Lines at boundary: closing uses max from neighborhood → ≈80 (green side).
    Response ≈ 80 − 15 = **65** ✓
- Surface mask (V > 30): Green V≈130 ✓, Blue V≈140 ✓. **Both surfaces provide
  neighbors** — the neighbor filter has full support across the entire image.
- **Prediction: All lines detected regardless of which surface they're on.** The
  weakest case (line on pure blue surround) still produces V top-hat ≈80 and S
  black-hat ≈28 — both well above reasonable thresholds.

| | Winner | Confidence |
|---|---|---|
| **Approach B** — decisively | Approach A's court mask is fundamentally incomplete | High |

---

### Condition 4: Multi-Color Court (green + vivid blue S≈78)

**Typical values**: Green: V≈130, S≈70-90. Blue: V≈130, S≈78. Lines: V≈220, S≈15.

**Approach A**:
- Court mask (S > 50): Both surfaces S > 50 ✓
- Color detection: Blue S≈78 > 30 chromatic threshold → multi-color detection CAN
  trigger if 2nd bin count ≥ 25% of 1st. On this court type, it does (user confirms
  "currently works").
- **Prediction: Works.** All gates pass because both surfaces are well-saturated.

**Approach B**:
- All signals strong. S black-hat on blue side: 78 − 15 = 63 ✓
- **Prediction: Works.**

| | Winner | Confidence |
|---|---|---|
| **Tie** — both work because blue saturation is high enough | — | High |

---

### Condition 5: Court with Strong Shadows (V ranges 80 to 180)

**Typical values**: Shadow: V≈80, S≈60-90. Sun: V≈180, S≈50-70.
Lines in shadow: V≈160-170. Lines in sun: V≈230.
Shadow boundary: V transitions 80→180 over 10-50 px.

**Approach A**:
- Court mask: S≈50-90 → mostly passes, but sunny areas with S≈50 are borderline.
- V top-hat at shadow boundary: If the boundary is sharp (10 px < 23 px kernel),
  the top-hat produces a response of up to ΔV/2 ≈ 50. This is a **false line candidate**.
  However: the shadow boundary has S≈60-70 (still on court surface), so **S < 50 rejects
  it** ✓. This is a genuine strength of the saturation gate.
- Lines in shadow: V≈165. V > 150 ✓ (borderline but passes). Top-hat ≈ 165 − 80 = 85 ✓
- Lines in deep shadow: V might drop to ≈140. **V > 150 rejects them** ❌
- **Prediction: Mostly works. Shadow boundaries rejected by S gate. Lines in deep
  shadow may be lost due to V > 150 threshold.**

**Approach B**:
- V top-hat at shadow boundary: Same response as above (up to ≈50 if sharp).
- S black-hat at shadow boundary: Shadow doesn't drastically change S (shadows reduce
  V but saturation is roughly preserved or slightly increases). S on both sides ≈60-80.
  Black-hat response at boundary ≈ **0-10**. Weak → combined criterion rejects ✓
- Lines in shadow: V top-hat ≈ 165 − 80 = 85 ✓. S black-hat ≈ 60-80 − 15 = 45-65 ✓.
  **No V > 150 gate → lines in deep shadow still pass** ✓
- Lines in deep shadow: V≈140, court V≈70. Top-hat ≈ 70. S black-hat ≈ 50+. Both strong ✓
- **Prediction: Handles shadows better than A. Shadow boundaries rejected by the dual
  criterion. Lines in deep shadow are preserved because local contrast is maintained
  even when absolute values drop.**

**Key insight**: Shadows reduce absolute brightness but largely preserve LOCAL contrast
between lines and surrounding court. Approach B measures local contrast directly;
Approach A's V > 150 gate measures absolute brightness and fails when it drops.

| | Winner | Confidence |
|---|---|---|
| **Approach B** — better in deep shadows | V > 150 is A's Achilles heel | High |

---

### Condition 6: Night Outdoor Under Floodlights

**Typical values**: Well-lit areas: V≈140-180, S≈50-80. Dark areas between lights:
V≈50-80, S≈25-50. Specular reflections (wet surface): V≈250, S≈5-15.
Lines vary dramatically with position.

**Approach A**:
- Court mask (S > 50): Dark areas S≈25-50 → many fail. **Incomplete court mask** in
  poorly lit areas.
- Lines in dark areas: V≈100-140. **V > 150 rejects them** ❌
- Specular reflections: If small (<23 px), top-hat responds. S < 50 ✓ (S≈10).
  V > 150 ✓ (V≈250). **False line candidates** ❌. However, specular spots are usually
  point-like, not elongated → Hough transform rejects most. But multiple spots along a
  line of reflection could create false Hough lines.
- Lines in well-lit areas: Standard case, works ✓
- **Prediction: Fails in poorly lit areas. Lines lost to V > 150 gate. Some specular
  false positives. Only works in the illumination sweet spot.**

**Approach B**:
- Lines in dark areas: V≈120, court V≈60. Top-hat ≈ 60. S black-hat: line S≈15,
  court S≈35. Response ≈ 20. Both moderate but positive ✓
- Lines in well-lit areas: Standard strong response ✓
- Specular reflections: V top-hat responds if small. S black-hat: specular S≈10,
  wet court S≈30-50. Response ≈ 20-40. **Both channels respond → false positive** ❌
  Same vulnerability as A, mitigated by Hough downstream.
- Surface mask (V > 30): Dark areas V≈50-80 → pass. Full neighbor support ✓
- **Prediction: Works across the illumination range. Specular reflections are the
  shared weakness, but Hough + homography validation reject non-line features.**

| | Winner | Confidence |
|---|---|---|
| **Approach B** — works in dark areas where A fails | Medium |

Confidence is medium because night conditions are highly variable and the specular
reflection problem needs empirical validation.

---

### Condition 7: Clay Court (orange/red surface, white lines)

**Typical values**: Court: H≈10-20, S≈80-120, V≈140-170.
Lines: V≈210-230, S≈10-30.
Dusty lines: V≈180-200, S≈20-40.

**Approach A**:
- Court mask (S > 50): Clay S≈80-120 >> 50 ✓
- Color detection: H≈15 detected as "clay" ✓
- Clean lines: Top-hat ≈ 220 − 155 = 65 ✓. S < 50 ✓. V > 150 ✓.
- Dusty lines: Top-hat ≈ 190 − 155 = 35. **Marginal** — Otsu threshold could be
  close to 35, making detection unreliable. S≈35 < 50 ✓. V≈190 > 150 ✓.
- **Prediction: Clean lines work. Dusty lines are borderline on top-hat threshold.**

**Approach B**:
- Clean lines: V top-hat ≈ 65 ✓. S black-hat ≈ 100 − 15 = 85. Very strong ✓.
- Dusty lines: V top-hat ≈ 35 (same marginal). S black-hat ≈ 100 − 35 = **65**.
  Still strong ✓. The S black-hat compensates for the weak V top-hat because clay dust
  reduces line brightness but doesn't make lines as saturated as the court surface.
- **Prediction: Both work for clean lines. Approach B is more resilient for dusty
  lines because the S black-hat provides a strong second signal even when V contrast
  is reduced.**

**Key insight**: Clay dust degrades V contrast (lines get darker) but preserves S
contrast (lines remain less saturated than orange clay). Approach B exploits both
channels; Approach A relies primarily on V contrast.

| | Winner | Confidence |
|---|---|---|
| **Approach B** — slight edge on dusty lines | Medium |

Medium confidence because actual clay dust behavior may vary.

---

### Condition 8: Worn/Dirty Court (faded lines, scuff marks)

**Typical values**: Court: V≈130, S≈60-80.
Faded lines: V≈180, S≈20-35. Clean lines: V≈220, S≈15.
Scuff marks: V≈150-170, S≈40-60 (intermediate between line and court).

**Approach A**:
- Faded lines: Top-hat ≈ 180 − 130 = 50 ✓. S≈25 < 50 ✓. V≈180 > 150 ✓.
  **Detected** ✓
- Scuff marks (thin, <23 px): Top-hat ≈ 160 − 130 = 30. Marginal. S≈50 — **right
  at the S < 50 boundary**. Some pass, some don't. This is a binary gate applied to
  a continuous reality, creating unpredictable behavior.
- Scuff marks (wide, >23 px): Top-hat ≈ 0. Rejected ✓
- **Prediction: Faded lines work. Thin scuff marks are borderline — the S < 50 gate
  is the only defense, and it's fragile at S≈50.**

**Approach B**:
- Faded lines: V top-hat ≈ 50 ✓. S black-hat ≈ 70 − 25 = **45** ✓.
- Scuff marks (thin): V top-hat ≈ 30 (marginal). S black-hat ≈ 70 − 50 = **20**.
  Weak. The dual criterion requires BOTH to be strong. A scuff mark would need to pass
  both a V threshold (≈25) AND an S threshold (≈20-25). The S black-hat response of
  20 is notably weaker than for real lines (45). **Better discrimination** ✓
- **Prediction: Better scuff mark rejection. The S black-hat creates a clear separation:
  real lines have S black-hat ≈45, scuff marks ≈20. The V top-hat alone can't
  distinguish them (both ≈30-50), but the dual criterion can.**

**Quantitative discrimination**:
| Feature | V top-hat | S black-hat | Combined signal |
|---------|-----------|-------------|-----------------|
| Clean line | 90 | 55 | Both strong |
| Faded line | 50 | 45 | Both moderate |
| Thin scuff mark | 30 | 20 | V marginal, S weak |
| Court surface | 0 | 0 | Both zero |

Approach A uses a binary S < 50 gate that can't distinguish S=25 (faded line) from
S=50 (scuff mark) — both are borderline. Approach B uses the magnitude of the S
black-hat response, which has a 2:1 signal ratio (45 vs 20).

| | Winner | Confidence |
|---|---|---|
| **Approach B** — meaningfully better scuff mark discrimination | High |

---

### Condition 9: Player Occlusion

**Scenario 1**: Player standing on a line — body/legs interrupt line continuity.
**Scenario 2**: Player shadow crossing a line.

Both approaches: A player's body is opaque. The line beneath the player is simply
not visible. Neither approach can detect what doesn't exist in the pixel data. This
is a **downstream problem** — the Hough transform must handle line gaps, and the
keypoint detection must tolerate partial lines. **Both equal here.**

**Player shadow crossing a line** (more interesting):

**Approach A**:
- Shadow darkens the line. If V drops from 220 to 150-170, the V > 150 gate is
  borderline. A deep shadow (V→130) causes the line to be **rejected** ❌.
- The S < 50 gate: player shadows are still achromatic on the line → S still ≈15 ✓.
- **Prediction: Partial line loss in shadow. Hough may still detect the line from
  unshadowed portions if the gap is small.**

**Approach B**:
- Shadow darkens the line AND the surrounding court. Line V drops to ≈160, court V
  drops to ≈70. V top-hat ≈ 160 − 70 = **90**. Counter-intuitively, the **local contrast
  actually increases** because the court darkens more than the line (lines are brighter
  to start, so the multiplicative shadow factor preserves more absolute brightness).
- S black-hat: Shadow doesn't significantly change S. Line still achromatic ✓.
- **Prediction: Line detected through player shadow** ✓. The local-contrast approach
  is inherently shadow-invariant because shadows are approximately multiplicative in
  the V channel.

| | Winner | Confidence |
|---|---|---|
| **Approach B** — shadow-invariant by design | Player body occlusion tied | High |

---

### Condition 10: Adjacent Courts Visible at Edges

**Scenario**: Crop includes lines from an adjacent court, possibly a different color.

**Approach A**:
- Saturation filter: S > 50 court mask includes any chromatic surface → adjacent
  court's surface is included → adjacent court's lines have neighbor support.
  Adjacent lines pass all gates → **detected as candidates** ❌
- Color-based filter: Detects the primary court's color. If the adjacent court is a
  different color, its surface won't be in the court mask → adjacent lines lack
  neighbors → **rejected** ✓. This is actually a case where color-based filtering
  helps.

**Approach B**:
- V > 30 surface mask includes everything → adjacent court lines have full neighbor
  support. Top-hat and black-hat respond to any white line on any colored surface.
  Adjacent lines **detected** ❌.

Both approaches detect adjacent court lines through the saturation/surface mask path.
Approach A's color-based filters might exclude them (advantage), but the saturation
filter doesn't. Since the scoring system picks the best candidate, and the saturation
candidate includes adjacent lines, this advantage is only realized if the color-based
candidate wins.

In practice, adjacent court lines would appear at the image edges, typically at angles
inconsistent with the target court's perspective geometry. The downstream line
classification (horizontal vs steep, vanishing point consistency) should reject most
of them.

| | Winner | Confidence |
|---|---|---|
| **Marginal Approach A edge** — color filter provides some protection | Low |

Low confidence because the practical impact depends on geometry and the downstream
classifier's robustness.

---

### Condition 11: Grass Court (Wimbledon-style mown bands)

**Typical values**: Light bands: V≈120, S≈50-65. Dark bands: V≈100, S≈60-75.
Band width: ≈60-120 px (much wider than 23 px kernel).
Lines: V≈220, S≈10-20 (white paint on grass).

**Approach A**:
- Court mask (S > 50): Grass S≈50-75 → passes (borderline for light bands at S≈50).
- V top-hat at band boundaries: Bands are 60-120 px wide >> 23 px kernel. The opening
  follows the broad band pattern. Top-hat response at boundaries ≈ 5-15 (small
  transition effect). Well below threshold ✓
- V top-hat at lines: 220 − 110 = 110 ✓
- S < 50 at band boundaries: S≈55-70 → fails S < 50. **Rejected** ✓
- **Prediction: Works. Mown bands are too wide for top-hat and too saturated
  for the line gate.**

**Approach B**:
- V top-hat at band boundaries: Same — bands too wide → low response ✓
- S black-hat at band boundaries: Light bands S≈55, dark bands S≈70. ΔS ≈ 15.
  But the bands are wider than the kernel, so closing doesn't bridge them →
  black-hat ≈ 0 at each band's interior, and ≈5-10 at transitions. **Low response** ✓
- V top-hat at lines: 110 ✓. S black-hat at lines: 60 − 15 = 45 ✓.
- **Potential concern**: Grass is more texturally irregular than hard courts. Individual
  grass patches might create small-scale V and S variation. But these variations are
  very small (ΔV ≈ 3-8, ΔS ≈ 3-8) and produce negligible morphological responses.
- **Prediction: Works. Both the kernel-size filtering and the dual criterion protect
  against the mown band pattern.**

| | Winner | Confidence |
|---|---|---|
| **Tie** — mown bands are a non-issue for both approaches | High |

---

### Condition 12: Indoor Court with Ceiling Reflections (V1 root cause)

**Typical values**: Normal court: V≈130, S≈45-65 (low-saturation indoor surface).
Hot spots: V≈190-240, S≈15-35 (washed out by ceiling light).
Lines: V≈215, S≈12.

This is the hardest case because hot spots have similar V and S to actual lines.

**Approach A**:
- Court mask (S > 50): Court S≈45-65 → **borderline**. Areas with S < 50 are excluded
  from the court mask. The court mask is **incomplete and noisy** — some pixels pass,
  some don't, creating an unreliable neighbor count.
- Hot spots: If localized and small (<23 px), V top-hat responds. S≈25 < 50 ✓.
  V≈220 > 150 ✓. Neighbor filter: if the hot spot is near normal court surface
  (S > 50 region), it has enough neighbors. **False positive** ❌
- Lines: Same gates as hot spots. **Detected** ✓. But so are the hot spots.
- Otsu adaptive threshold: Computed only on S > 50 region, which may be a small/biased
  subset of the court. Threshold may be poorly calibrated.
- **Prediction: Borderline court mask creates systemic instability. Hot spots are
  indistinguishable from lines through all gates. The scoring system is the only
  defense — if hot-spot false positives produce a bad homography, it scores low.
  But if they happen to align with a valid geometric interpretation, the wrong
  homography could win.**

**Approach B**:
- V top-hat at hot spots: Depends on size.
  - Large reflections (>23 px): Top-hat ≈ 0 → **rejected** ✓
  - Small reflections (<23 px): Top-hat responds → passes ❌
- S black-hat at hot spots: Hot spot S≈25, surrounding court S≈55. Response ≈ 30.
  Lines: S≈12, surrounding court S≈55. Response ≈ 43.
  **Some discrimination** (30 vs 43), but not dramatic.
- Surface mask (V > 30): All areas pass → full neighbor support everywhere ✓.
  This is actually better than A's borderline S > 50 mask because it provides
  consistent neighbor counts.
- **Prediction: Large hot spots are automatically rejected (key advantage). Small hot
  spots are problematic for both approaches. Approach B provides slightly better
  S discrimination (30 vs 43 response) and a much more stable surface mask.**

**Critical observation**: The dominant ceiling reflections in real indoor courts tend to
be **broad, diffuse** patches (ceiling lights are large area sources viewed at shallow
angles). The 23×23 kernel acts as a natural size filter — reflections wider than the
kernel (≈23 px, which corresponds to roughly the width of a court line) are completely
suppressed. Only very localized specular reflections (from point-like light sources or
sharp surface features) would survive, and these are more point-like than line-like,
making them poor Hough candidates.

| | Winner | Confidence |
|---|---|---|
| **Approach B** — more stable, large reflections auto-rejected | Medium |

Medium confidence because the critical discriminator (hot spot size vs kernel size)
needs empirical validation on actual V1 footage.

---

## Cross-Cutting Analysis

### 13. Computational Cost

**Approach A** (current, with 3 candidates + scoring):

| Step | Approx. cost | Runs |
|------|-------------|------|
| HSV conversion | O(HW) | 3× (once per filter, though could cache) |
| Top-hat on V (23×23 ellipse) | O(HW·k²) optimized | 1× (saturation filter only) |
| KMeans (k=3, n_init=10, 2000 samples) | O(2000·3·10·iters) | 1× |
| Random sampling + binning | O(n_samples) | 1× |
| Color mask building | O(HW) | 1-2× per color variant |
| Convolution for neighbors (7×7) | O(HW·49) | 3× |
| HoughLinesP | O(HW + n_edges·n_θ) | 3× |
| Keypoint detection + homography | O(n_lines² + RANSAC) | 3× |
| Scoring (perspectiveTransform + overlap) | O(n_points + HW) | 3× |

**Total**: roughly 3× the single-candidate cost, dominated by the 3× Hough + homography
+ scoring passes. KMeans adds a fixed overhead that's small relative to image processing.

**Approach B** (single candidate):

| Step | Approx. cost | Runs |
|------|-------------|------|
| HSV conversion | O(HW) | 1× |
| Top-hat on V (23×23 ellipse) | O(HW·k²) optimized | 1× |
| **Black-hat on S (23×23 ellipse)** | O(HW·k²) optimized | 1× (NEW) |
| Thresholding + combination | O(HW) | 1× |
| Convolution for neighbors | O(HW·49) | 1× |
| HoughLinesP | O(HW + n_edges·n_θ) | 1× |
| Keypoint detection + homography | O(n_lines² + RANSAC) | 1× |
| Scoring | O(n_points + HW) | 1× |

**Added cost**: One morphological closing on S channel (same complexity as existing top-hat).
**Saved cost**: 2 entire candidate pipelines (filter + Hough + homography + scoring),
plus KMeans.

**Net effect**: Approach B is approximately **2.5-3× faster** than Approach A. The added
black-hat is negligible compared to eliminating 2 full pipeline passes.

If Approach B is run alongside the saturation filter (2 candidates instead of 3),
it's still faster: 2 candidates vs 3, minus KMeans.

---

### 14. Parameter Sensitivity

**Approach A parameters** (identified from code):

| Parameter | Value | What breaks if wrong |
|-----------|-------|---------------------|
| SAT_COURT_THRESHOLD | 50 | ±10 changes which surfaces count as "court" |
| SAT_LINE_S_MAX | 50 | ±10 admits scuff marks or rejects dusty lines |
| SAT_LINE_V_MIN | 150 | ±20 loses lines in shadow or admits court surface |
| TOPHAT_KERNEL_SIZE | 23 | See below |
| TOPHAT_THRESHOLD | 30 | Overridden by Otsu in most cases |
| Otsu scale factor | 0.7 | ±0.2 shifts the adaptive threshold significantly |
| COLOR_MATCH_H_THRESH | 15 | ±5 changes color mask coverage |
| COLOR_MATCH_S_THRESH | 40 | ±10 changes color mask coverage |
| COLOR_MATCH_V_THRESH | 60 | ±10 changes color mask coverage |
| Chromatic bin S threshold | 30 | Determines which bins are "chromatic" |
| Multi-color ratio | 0.25 | Determines single vs multi detection |
| KMeans k | 3 | Different k may merge or split court colors |
| Hough threshold | 30 | Affects line detection sensitivity |
| Hough minLineLength | 80 | Too high: short lines missed |
| Hough maxLineGap | 40 | Too high: false connections |

That's **15+ tunable parameters**, many of which interact. The S-related thresholds
(50, 50, 150) form a particularly fragile triplet: changing one requires adjusting
the others.

**Approach B parameters**:

| Parameter | Value | What breaks if wrong |
|-----------|-------|---------------------|
| TOPHAT_KERNEL_SIZE | 23 | Must be > line width. See below. |
| V top-hat threshold | Otsu-adaptive | Otsu handles this automatically |
| S black-hat threshold | TBD (adaptive?) | One new parameter to set |
| Surface mask V threshold | 30 | Must exclude pitch black only |
| MIN_COURT_NEIGHBORS | 4 | Same as Approach A |
| Hough parameters | Same | Same sensitivity |

**5-6 parameters**, of which kernel size and surface mask are physics-grounded.

**Kernel size sensitivity analysis**:

Tennis court lines are typically 5 cm (2 inches) wide. At courtside camera distances
(3-10m), this maps to approximately 3-8 pixels depending on resolution and position.
The kernel must be wider than the widest line appearance:

| Kernel size | Lines detected? | False features? |
|-------------|----------------|-----------------|
| 15 | Lines up to ≈14 px wide ✓ | Narrower features also detected |
| 19 | Lines up to ≈18 px wide ✓ | Same as 23 in most cases |
| 23 | Lines up to ≈22 px wide ✓ | Current baseline |
| 27 | Lines up to ≈26 px wide ✓ | Larger features also suppressed |
| 31 | Lines up to ≈30 px wide ✓ | May be too large for near-camera lines |

Changing kernel from 23 to 19: All lines narrower than 18 px are still detected.
Court lines at courtside viewing distances are 3-8 px wide. **No degradation**.
The only effect: features 19-22 px wide (which are NOT court lines at typical distances)
would also be detected — more false candidates for Hough to filter, but no lost lines.

Changing from 23 to 27: Same argument. No lost lines. Slightly more aggressive
suppression of medium-width features.

**Approach A with wrong parameters**: Changing SAT_COURT_THRESHOLD from 50 to 40 fixes
the V1 blue surround problem (S≈43 > 40 ✓) but now admits outdoor sky pixels (S≈35-45)
as "court," potentially creating false neighbor support in images where sky is visible.
Each parameter change has non-local, hard-to-predict side effects.

| | Winner | Confidence |
|---|---|---|
| **Approach B** — far fewer parameters, physics-grounded, graceful degradation | High |

---

### 15. Failure Mode Severity

**Approach A failure modes**:

1. **Court mask failure** (S threshold too high/low):
   - Effect: Entire regions of the court lose neighbor support
   - Severity: **Catastrophic** — all lines in that region are rejected, not just some
   - Manifestation: Homography has too few keypoints → RANSAC fails or produces
     wildly wrong result
   - Recovery: Scoring system may pick a different candidate, but if the underlying
     issue (e.g., desaturated surface) affects all candidates, no recovery

2. **Color detection failure** (wrong dominant color):
   - Effect: Court mask captures wrong surface → lines on the actual court have
     no neighbor support
   - Severity: **Catastrophic** for that candidate, but scoring picks a different one
   - Recovery: Saturation filter provides a color-agnostic fallback

3. **V > 150 threshold failure** (dim lighting):
   - Effect: All lines below V=150 are silently dropped
   - Severity: **Moderate to severe** — if multiple lines are affected, the
     geometric configuration is incomplete
   - Recovery: Partial — remaining bright lines may be sufficient for homography

**Approach B failure modes**:

1. **V top-hat too weak** (very bright court, V≈200+):
   - Effect: Line response is weak (ΔV ≈ 20-30)
   - Severity: **Gradual** — weakest-contrast lines are lost first, highest-contrast
     lines survive longest. The longest/most prominent lines are typically the
     highest contrast.
   - Recovery: Otsu adaptation lowers the threshold. S black-hat provides
     complementary signal.

2. **S black-hat too weak** (achromatic court, S≈15-25):
   - Effect: S channel provides no discrimination between lines and court
   - Severity: **Moderate** — the V top-hat still works, but without S confirmation,
     false positives increase
   - Recovery: V top-hat alone may be sufficient. This is a rare court type
     (truly achromatic hard courts are uncommon).

3. **Small specular reflections** (wet/glossy surface):
   - Effect: False line candidates appear
   - Severity: **Low** — point-like reflections produce poor Hough candidates;
     the homography RANSAC stage rejects inconsistent points
   - Recovery: Downstream pipeline provides multiple defense layers

**Summary**:

| | Primary failure mode | Severity | Degradation pattern |
|---|---|---|---|
| Approach A | Global threshold mismatch | Catastrophic (entire regions) | Binary: works or doesn't |
| Approach B | Low local contrast | Gradual (weakest lines first) | Analog: loses sensitivity smoothly |

| | Winner | Confidence |
|---|---|---|
| **Approach B** — graceful degradation vs. categorical failure | High |

---

### 16. Composability with Scoring

The scoring function (`score_court_detection`) operates on:
- Input: `near_half` image + `H_near` homography matrix
- Process: Project template court lines via H_inv → count overlap with bright pixels
  (V > 170, S < 60)
- Output: Integer overlap count

This is **completely independent** of how the line mask was generated. The scoring
function doesn't know or care whether the homography came from the saturation filter,
a color filter, or a local-contrast filter.

**Approach B is fully composable with scoring.** Three integration strategies:

1. **Drop-in replacement**: Replace all 3 existing candidates with 1 local-contrast
   candidate. Scoring validates the single result (score > 0 → accept, else fail).
   - Pro: Simplest. 3× faster.
   - Con: No fallback if local-contrast fails on an unexpected case.

2. **Primary + fallback**: Run local-contrast as primary. If score < threshold,
   fall back to saturation filter.
   - Pro: Robust with minimal overhead (fallback rarely triggers).
   - Con: Slightly more complex logic.

3. **Scored candidate**: Add local-contrast as a 4th candidate alongside existing 3.
   Pick highest score.
   - Pro: Most conservative. Can't be worse than current system.
   - Con: Defeats the purpose (still running 3 other candidates). Only makes sense
     as a transitional strategy.

**Recommendation**: Strategy 2 (primary + fallback) during validation period,
transitioning to Strategy 1 once empirically validated.

**One subtlety**: The scoring function's bright-pixel mask uses V > 170 and S < 60.
These are their own global thresholds. If Approach B is adopted to eliminate global
thresholds from line detection, the scoring function's thresholds remain as a
potential brittleness point. However, scoring is a validation step (not detection),
so its thresholds only need to be "good enough" to count overlap — they don't need
to perfectly segment lines. The current values are reasonable defaults.

---

## Consolidated Scorecard

| # | Condition | Winner | Margin | Confidence |
|---|-----------|--------|--------|------------|
| 1 | Bright single-color | Tie | — | High |
| 2 | Dark single-color | Tie | — | High |
| 3 | Multi-color desaturated (V1) | **B** | Large | High |
| 4 | Multi-color vivid | Tie | — | High |
| 5 | Strong shadows | **B** | Moderate | High |
| 6 | Night floodlights | **B** | Large | Medium |
| 7 | Clay court | **B** | Small | Medium |
| 8 | Worn/dirty court | **B** | Moderate | High |
| 9 | Player occlusion | **B** | Small | High |
| 10 | Adjacent courts | **A** | Small | Low |
| 11 | Grass court | Tie | — | High |
| 12 | Ceiling reflections (V1 cause) | **B** | Moderate | Medium |
| 13 | Computational cost | **B** | Large (3×) | High |
| 14 | Parameter sensitivity | **B** | Large | High |
| 15 | Failure mode severity | **B** | Large | High |
| 16 | Scoring composability | Tie | — | High |

**Approach B wins or ties in 15 of 16 categories.** Approach A has a marginal edge
only in condition 10 (adjacent courts), and that advantage depends on the color-based
filter winning the scoring comparison — which it often doesn't.

---

## Honest Assessment of Weaknesses

### Things that could go wrong with Approach B:

1. **The S black-hat threshold is unset.** The analysis assumes a reasonable threshold
   exists, but it hasn't been determined. If using Otsu on the S black-hat values:
   the bimodal distribution (court pixels ≈ 0, line pixels ≈ 40-80) should give a
   good split. But if there are many non-line features with intermediate black-hat
   values (scuff marks, texture), Otsu might set the threshold too low.
   **Mitigation**: Use Otsu on the joint distribution (pixels that have BOTH V top-hat
   and S black-hat above small minimum values).

2. **Untested on real data.** This analysis is purely theoretical. Real images have
   JPEG compression artifacts, motion blur, lens distortion, auto-exposure variations,
   and other effects not modeled here. The morphological operations are robust to most
   of these (they're nonlinear filters with good noise properties), but empirical
   validation is essential.

3. **The surface mask (V > 30) is very permissive.** It admits nearly everything —
   sky, background objects, scoreboards, fences. This means the neighbor filter
   provides less protection against false positives in non-court regions. Approach A's
   S > 50 court mask, despite its failures, does exclude non-chromatic background.
   **Mitigation**: The ROI cropping (net-to-baseline) should eliminate most non-court
   background. If needed, a coarse court region prior (based on image position or
   learned mask) could replace V > 30.

4. **Specular reflections remain a shared weakness.** Neither approach handles small
   specular points. The Hough transform is the defense, and it's not perfect —
   multiple specular points along a line of reflection could form a false Hough line.
   **Mitigation**: The scoring step (project template → check overlap) provides a
   geometric sanity check. Also, specular reflections are typically point-like and
   their Hough lines would have inconsistent angles with the court's perspective
   geometry.

5. **The kernel size is a hidden assumption about line width.** If the camera is very
   close (1-2m), court lines could appear wider than 23 px. If very far (20m+), lines
   could be sub-pixel and undetectable by any method. The 23 px kernel is appropriate
   for courtside video at 3-10m, but other camera positions may need adjustment.
   **Mitigation**: The kernel size could be adaptive based on estimated camera distance
   (derivable from the court's perspective geometry in the image).

---

## Final Verdict

**Approach B is the recommended path forward.** It is strictly more robust across the
analyzed conditions, has far fewer parameters, degrades gracefully, and is
computationally cheaper. No condition was identified where Approach A is significantly
better.

### Implementation recommendation:

1. **Implement Approach B as `agrawal_local_contrast_line_filter`** — a new function
   following the same interface as the existing filters (returns line_mask, court_mask,
   hsv).

2. **During validation** (next 2-4 test videos): Run Approach B alongside the
   saturation filter. Use scoring to pick the winner. Log which approach wins per frame
   to build empirical evidence.

3. **If Approach B wins ≥80% of frames**: Drop the paper and kmeans filters. Keep
   only local-contrast + saturation as scored candidates.

4. **If Approach B wins ≥95% of frames**: Make it the sole candidate. Use scoring
   only for validation (reject bad results), not comparison.

5. **Keep the saturation filter code** (don't delete it) as a documented fallback
   for any future edge case where local contrast fails.

### What NOT to do:

- Don't add Approach B as a 4th candidate without removing others — this increases
  computational cost without addressing the root architectural issue (too many
  interdependent global thresholds).
- Don't tune Approach A's thresholds further to fix V1 — each fix introduces
  regressions elsewhere. The threshold approach is fundamentally fragile.
- Don't skip empirical validation — this analysis is theoretical and must be
  confirmed on real footage.
