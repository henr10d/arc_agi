tohle je task073 na vstupu je vzdy 5x5 input kde ve treti rade jsou modre kostky ktere jsou posunute na spodni radek. Dej mi prompt pro vytvoreni onnx s nejlepsim score
Create the smallest valid ONNX model for NeuroGolf task073.

Task rule:
- Input grid is always 5x5.
- Output grid is also 5x5.
- Copy the input unchanged except:
  - Any blue cells in row index 2 / third row must be moved vertically down to the bottom row, same column.
  - The original blue cells in row 2 become black/background.
  - Existing bottom row gray cells stay unchanged except where overwritten by moved blue.
- In other words: output[4, c] = blue wherever input[2, c] is blue; output[2, c] = 0 for those blue positions; all other cells copied.

Optimize only for NeuroGolf score:
- minimize internal tensor memory + initializer params
- do not optimize ONNX file size unless it reduces score cost
- graph input/output tensors are free; internal tensors are expensive
- avoid full one-hot expansion if possible
- avoid large intermediate tensors
- prefer simple Slice/Gather/Scatter/Where/Concat logic over big masks
- any opset is allowed if it lowers cost

Implementation target:
- Build `task073.onnx`
- Input shape should match NeuroGolf expected ARC tensor format `[1,10,30,30]`.
- Only the active 5x5 area matters.
- Extract top-left 5x5/channel-format grid.
- Produce corrected 5x5 grid, then pad/embed back to required output shape `[1,10,30,30]`.
- Use color id 1 for blue, 0 for background/black, 5 for gray if needed, but avoid hardcoding gray unless necessary.

Important:
- Try multiple graph constructions and measure NeuroGolf score.
- Keep the best-scoring valid model.
- Report memory, params, cost, and score.

------------------------------------------------
tohle je task 74 na vstupu je nejaky pattern s cervenyma blokama na vystupu je patern doplnen bez cervenych bloku. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score.
Přemýšlení několik sekund

Použij tenhle prompt pro Codex/Cursor:

We need solve NeuroGolf ARC task074 with an optimized ONNX model.

Task:
Input/output are always 30x30 one-hot ARC tensors [1,10,30,30].
The input contains a mostly regular colorful repeating/radial/diagonal pattern, but some cells are corrupted by solid red blocks/color 2.
The expected output is the same completed pattern with all red corruption removed/replaced by the correct underlying pattern color.

Goal:
Create the smallest/highest-score ONNX possible. Optimize NeuroGolf score, not readability or file size alone.
Do not force a specific opset. Use any ONNX opset/operators that reduce memory + params.

Important observations:
- Red color 2 in the input is not part of the real pattern; it marks occlusion/corruption.
- Output preserves all non-red cells from input.
- Only red cells need to be replaced.
- The missing values can be reconstructed from the surrounding periodic/symmetric pattern.
- Pattern appears globally structured: mirrored / repeated diagonals and local periodic motifs across the 30x30 grid.
- Need infer replacement color for each red cell from nearby non-red cells at same relative pattern phase.
- If several training examples exist, exploit that the corruption color is always 2 and true pattern colors are copied from matching positions.

Implementation strategy:
1. Extract class grid from one-hot input using ArgMax over channel axis.
2. Build mask_red = (grid == 2).
3. For each red cell, predict replacement by looking at candidate cells with the same pattern phase:
   - nearby left/right/up/down non-red cells,
   - diagonal symmetric counterparts,
   - periodic offsets likely useful for 30x30: ±1, ±2, ±3, ±5, ±10, ±15 in x/y and diagonal combinations.
4. Prefer exact local reconstruction:
   - For every position, collect candidate colors from shifted versions of the grid.
   - Ignore candidate values equal to 2 and 0 when possible.
   - Use deterministic priority ordering: closest valid neighbor / symmetric counterpart first.
5. Build output_grid = Where(mask_red, reconstructed_color, grid).
6. Convert output_grid back to one-hot [1,10,30,30].
7. Ensure graph output shape is exactly [1,10,30,30].

Optimization constraints:
- Avoid large internal tensors where possible.
- Internal [1,10,30,30] tensors are expensive; prefer [1,1,30,30] int/bool masks and only one final one-hot if needed.
- Avoid Conv kernels with many params unless they reduce memory enough.
- Use Slice/Pad/Concat/Gather/Where/Equal/Cast/ArgMax/OneHot style ops.
- Constants should be tiny scalars or small index tensors only.
- Measure score after each candidate.
- Keep the best ONNX by actual NeuroGolf score.

Try multiple versions:
A) Minimal baseline:
   - output = input with red channel zeroed and replaced from nearest non-red shifted candidates.
B) Symmetry/periodic version:
   - candidate replacement from mirrored/periodic positions around center of 30x30.
C) Hybrid:
   - priority chain of shifts: horizontal/vertical/diagonal small offsets first, then ±5/±10/±15 periodic offsets.

Deliver:
- A Python script that generates `solution.onnx`.
- The script should be self-contained with hardcoded paths if needed.
- After generation, run ONNX checker, shape inference, and NeuroGolf scoring.
- Print memory, params, cost, score.
- Iterate until no simpler/higher-score graph is found.

Nejdůležitější je netvořit velké 10kanálové mezivýsledky. Drž rekonstrukci co nejdéle jako [1,1,30,30] grid integerů a one-hot udělej až úplně na konci.
----------------------------------------------------

tohle je task 75  vypada to ze vstup je vzdy 9x13 jde o to ze v levem hornim rohu je patern a ten je zkopirovan na mista modrych tecek. Dej mi prompt pro vytvoreni co nejefektivnejsiho onnx s maximalnim moznim score.

Prompt pro Codex/Cursor:

We need solve NeuroGolf ARC task075 with the smallest/highest-score ONNX possible.

Task:
- Input/output shape is always 9x13.
- Colors are ARC integer colors, one-hot model input/output expected by the existing NeuroGolf harness.
- Top-left contains a 3x3 colored pattern in rows 0..2, cols 0..2.
- Column 3 is a vertical gray separator/marker and must remain unchanged.
- Blue cells (color 1) mark target copy positions.
- Output copies the 3x3 pattern to each blue marker position, replacing the blue cell and its local 3x3 area.
- In the shown example:
  - source pattern = input[0:3, 0:3]
  - first copy target has blue at row 1 col 5 -> paste pattern at rows 0..2, cols 4..6
  - second blue at row 4 col 8 -> paste pattern at rows 3..5, cols 7..9
  - third blue at row 7 col 8 -> paste pattern at rows 6..8, cols 7..9
- In general, each blue marker is the center cell of a 3x3 paste area. Paste source pattern with top-left at `(blue_row - 1, blue_col - 1)`.
- Preserve all original non-overwritten cells, including the gray vertical separator.
- Blue markers disappear because they are overwritten by the pasted source pattern.

Optimization goal:
- Maximize NeuroGolf score, not readability.
- Minimize internal tensor memory and initializer scalar count.
- Input/output tensors are free; internal tensors are expensive.
- Avoid building large [1,10,9,13] intermediate tensors unless absolutely necessary.
- Prefer tiny slices, gathers, reshapes, one-hot channel slices, and direct concat/where construction.
- Do not optimize ONNX file size unless it reduces counted params or internal memory.

Implementation idea:
1. Extract source patch `P = X[:, :, 0:3, 0:3]`.
2. Detect blue markers from channel 1 only, ideally only in valid possible center locations.
3. Because grid is fixed 9x13 and paste blocks align as 3x3 blocks in this task, construct output by fixed 3-row bands:
   - rows 0..2: left original/source area + separator + paste if blue center exists in row 1 at candidate centers.
   - rows 3..5: preserve cols 0..3, paste P around blue centers in this band.
   - rows 6..8: preserve cols 0..3, paste P around blue centers in this band.
4. The cheapest robust route is likely to create candidate masks for 3x3 block top-left positions from blue channel:
   - candidate center rows: 1, 4, 7
   - candidate center cols probably 5, 8, 11 or any valid center 1..11 if needed.
   - mask scalar `m = X_blue[center_r, center_c]`
   - block = `Where(m, P, original_slice_for_that_3x3)`
5. Assemble final 9x13 with Concat from small 3x3/3x1/3x? slices.
6. If candidate columns are fixed to groups after separator, hardcode them for lower score cost:
   - source/source-copy area: cols 0..2
   - gray separator: col 3
   - target blocks: cols 4..6, 7..9, 10..12
   - center cols: 5, 8, 11
   - center rows: 1, 4, 7
   This gives a clean 3x4 block grid: `[pattern][gray col][slot0][slot1][slot2]`.
7. For each slot block:
   - top-left block at rows `3*br : 3*br+3`, cols `4 + 3*bc : 4 + 3*bc+3`
   - marker center at `(3*br+1, 4+3*bc+1)`
   - if marker is blue, output P there, else keep original block.
   - For br=0, slot0 may contain an existing copy if marker there; top-left source block remains input pattern.

Expected behavior from example:
- P from cols 0..2 is copied into slot block cols 4..6 in top band because blue center at (1,5).
- P copied into rows 3..5 cols 7..9 because blue center at (4,8).
- P copied into rows 6..8 cols 7..9 because blue center at (7,8).

Create/generate an ONNX file for task075 and run the NeuroGolf validator/scorer. Iterate for score:
- First make a correct simple graph.
- Then reduce memory by replacing full-grid Where/masks with per-block scalar Where and Concat.
- Avoid Cast/Equal on full grid; slice only blue channel scalars.
- Keep constants minimal; use Slice starts/ends/axes constants only where needed.
- Measure params and internal memory after ONNX shape inference/profiling.
- Report exact memory, params, cost, and score.

------------------------------------------------
tohle je task 076 jde o to ze je nekde patern a na dalsich mistech je nedokonceny na outputu je vsude dokonceny. Dej mi prompt pro vytvoreni nejefektivnejsiho onnx s maximalne nejlepsim score
Přemýšlení sekundu

Tady je prompt pro Cursor/Codex:

You are optimizing NeuroGolf ARC task076.

Goal:
Create the most score-efficient ONNX model possible for this ARC task. Optimize for NeuroGolf score, not readability or ONNX file size.

Task:
Input and output are always 13x13 grids.

The input contains:
1. One complete reference pattern.
2. One or more partial/incomplete copies of the same pattern elsewhere.
3. The output keeps the original grid but completes every partial copy so that each occurrence becomes the full reference pattern.

Observed rule:
- The full pattern is a small multi-color shape.
- Partial copies contain enough anchor pixels to determine the same orientation/placement.
- Missing colored cells are filled in from the complete reference pattern.
- Background remains 0.
- Existing nonzero pixels are preserved.
- The output is the union of the original input and the reconstructed missing cells for every detected copy.

Example behavior:
- Detect the complete source pattern.
- Detect partial instances by matching subsets of that pattern.
- For every valid partial match, stamp the full pattern at that position.
- Return the completed 13x13 grid padded/encoded as required by the NeuroGolf harness.

Implementation strategy:
Build a static ONNX graph specialized for 13x13.

Prefer extremely cheap tensor operations:
- Slice / Gather / Unsqueeze / Cast / Equal / Where / Mul / Add / Or / Max.
- Avoid Conv unless it clearly reduces memory+params.
- Avoid large one-hot tensors if possible.
- Avoid dynamic shape-heavy logic.
- Avoid internal tensors larger than necessary.
- Inputs/outputs are free, internal tensors are expensive.

Recommended approach:
1. Extract the known complete pattern location from the training examples if it is fixed or can be inferred cheaply.
2. Represent the reference pattern as a compact set of relative offsets and colors.
3. For each possible 13x13 placement where the pattern could fit:
   - Check whether all nonzero pixels currently present at that candidate location are compatible with the reference pattern.
   - Check whether at least a minimal anchor subset exists, so random noise does not trigger stamping.
4. If candidate is valid, write/stamp all reference pattern pixels into output.
5. Combine all candidate stamps with original input using Where / Max, preserving existing colors.

Important optimization:
Do not construct a full `[num_candidates, 13, 13]` tensor if avoidable. Instead generate only the required output cells directly:
- For each output cell, determine if it belongs to any valid stamped pattern.
- Use boolean masks per candidate and combine with OR/Max.
- Keep candidate-valid masks scalar/small.
- Only materialize final 13x13 output.

Scoring objective:
Minimize:
- initializer scalar count,
- Constant tensor/list attributes,
- internal tensor memory after ONNX shape inference and ORT profiling.

Hard constraints:
- The model must be deterministic.
- Must solve all train/test examples for task076.
- Do not optimize only for the shown screenshot; infer the general task rule.
- Try multiple ONNX graph designs and measure actual NeuroGolf score.
- Keep the best-scoring valid ONNX.

Deliverables:
1. A Python script that builds the ONNX model.
2. A validation script that runs it on all available task076 examples.
3. A score report comparing alternatives.
4. Save the best model as `task076_best.onnx`.

Search aggressively for cheaper formulations:
- constant-offset stamping,
- candidate anchor checks,
- sparse pattern reconstruction,
- direct cell formulas,
- using Max/Where instead of Scatter if cheaper,
- reducing params by deriving colors from the reference pattern in input rather than storing them as constants.

Do not force a specific opset. Use whatever ONNX opset gives the best measured NeuroGolf score.

-------------------------------------------------------------

tohle je task077 jde o to ze na vstupu jsou dve barvy majoritni a minoritni minoritni barva je doplnena zlutou aby tvorila ctverec. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Zkopíruj do Cursor/Codex:

We need solve NeuroGolf ARC task077 as an optimized ONNX model.

Task rule:
Input/output are always 17x18 one-hot ARC grids.
There are many blue distractor pixels/blocks. The meaningful objects are two same-color rectangular-ish red regions. For each red region, complete its missing bounding square/rectangle using yellow. The red pixels stay red. Yellow is only placed in the missing cells needed to make the minor/incomplete red shape match a filled square/rectangle footprint.

Observed behavior:
- Detect red color = 2.
- Find connected red components.
- Each component represents a partial filled rectangle/square.
- For each red component, compute its bounding box.
- Fill every cell inside that bounding box that is not already red with yellow = 4.
- Keep all other input cells unchanged.
- Output shape stays [1,10,17,18] or required NeuroGolf padded format if the framework expects [1,10,30,30].

Optimization goal:
Create the smallest/cheapest valid ONNX, not clean code.
Minimize NeuroGolf cost = internal tensor memory + params.
Graph inputs/outputs are free, but internal tensors are expensive.
Avoid large [1,10,30,30] internal tensors.
Operate on compact [17,18] / [1,17,18] masks as long as possible.
Only pad/expand to required output shape at the very end if necessary.
Avoid initializers when constants can be Constant nodes, but also test both because params count matters.
Prefer integer/bool masks over float tensors where ORT supports it.

Implementation strategy:
1. Extract class planes from one-hot input:
   - red mask from channel 2
   - optionally keep original full grid for final output.
2. Build row/column projections of red mask.
3. Because there are two separated red components, avoid generic connected components if expensive.
   Instead derive bounding boxes using row/column red presence:
   - identify contiguous row groups containing red
   - identify contiguous column groups containing red for each row group
   - produce rectangle mask for each red component.
4. Rectangle fill mask = bbox_mask AND NOT red_mask.
5. Yellow output plane = original yellow OR fill_mask.
6. Red plane remains original red.
7. Other color planes remain unchanged.
8. If yellow overwrites black/blue inside bbox, expected output suggests yellow fills only missing cells of red rectangle; in examples those cells may replace whatever was there. So set class 4 on fill_mask and clear all other planes at fill_mask.
9. Final one-hot output:
   - where fill_mask: yellow
   - elsewhere: original input.
10. Pad to [1,10,30,30] only at final output if required.

Important:
Do not implement a huge generic ARC solver.
Do not use loops outside graph unless generating static ONNX constants is cheaper.
Try several ONNX versions and benchmark actual NeuroGolf score.
Report memory, params, cost, score for each candidate.

Candidate optimizations to test:
A. Pure mask/projection bbox completion.
B. Hardcoded 17x18 coordinate masks with ReduceMax over rows/cols.
C. Split top/bottom red components by row gaps, because task examples appear to have two red regions.
D. If all hidden tests preserve fixed positions/ranges, test a more aggressive specialized solution, but only keep it if it passes all train examples.

Deliverable:
A Python script that builds task077.onnx, runs it on all available examples, verifies exact output, profiles NeuroGolf cost, and prints score.
Focus on the lowest measured cost, not readability.


-------------------------------------------

tohle je task078 jde o to ze nahore jsou modre kostky a do le jsou cervene ktere jsou posunuti do nahoru nejvic jak to jde. Dej mi prompt pro vytvoreni onnx s nejvetsim moznim score

Použij tenhle prompt pro Codex/Cursor:

We need create an ultra-low-cost ONNX model for NeuroGolf ARC task078.

Task:
Input and output are always 10x10 one-hot/color grid tensors with 10 channels.
There are blue cells in the upper part and red cells in the lower part.
The transformation is gravity-like upward movement of red cells:
- Keep all blue cells exactly where they are.
- Move every red connected vertical/shape component upward as far as possible.
- Red cells stop when the next upward move would collide with a blue cell, another red cell/component, or the top boundary.
- Background stays 0.
- Output is still 10x10, same canvas size.

Observed example:
Blue obstacles occupy the top rows. Red blocks from bottom are shifted upward into the highest valid free positions below/around blue obstacles. The red shape/columns preserve their relative structure while being translated upward as much as possible.

Goal:
Build `solution.onnx` with maximum NeuroGolf score, minimizing:
1. internal tensor memory after ONNX shape inference / ORT profiling
2. scalar params in initializers and Constant tensors/lists

Do not optimize for ONNX file size. Optimize only NeuroGolf cost = memory + params.

Implementation strategy:
- Avoid large `[1,10,30,30]` internal tensors.
- Work only on the required 10x10 area internally.
- Prefer uint8/bool/int64 masks where possible.
- Extract blue mask and red mask from input channels.
- Compute target position for each red cell/column by compacting red upward subject to blue blockers.
- Since grid is fixed 10x10, hardcode row/column logic if it reduces cost.
- Use minimal ONNX ops: Slice/Gather/ReduceSum/Greater/Where/Mul/Add/Concat/Expand only if beneficial.
- Avoid convolution and large constants.
- Constants should be tiny scalars or small index vectors only when unavoidable.

Important:
Try multiple graph designs and measure actual NeuroGolf cost/score after ORT profiling. Keep the best measured model.
Prefer a slightly ugly hardcoded graph if it has lower memory/params.

Deliver:
- `solution.onnx`
- script that builds it
- small verifier using the provided train examples
- print NeuroGolf memory, params, cost, score

--------------------------------

tohle je task 079 jde o to ze na vstupu jsou nejake paterny ruzne barvy  output je tan patern ktery se tam objevi nejvicekrat. vypada to ze vstup je vzdy 14x14. Dej mi prompt pro vytvoreni onnx s nejvetsim moznim score 
Task079 NeuroGolf / ARC-AGI ONNX optimization prompt:

We need build the highest-score ONNX solver for task079.

Task:
- Input is always 14x14 ARC grid.
- Output is 3x3.
- Input contains several small 3x3-like colored patterns in different colors.
- The correct output is the shape/pattern that appears the most times in the input, normalized to its own color.
- In the shown example, the olive/yellow pattern appears 4 times, while gray/red appear once, so output is the 3x3 olive pattern.

Important:
- Optimize for NeuroGolf score, not readability.
- Score uses `cost = internal_tensor_memory + params`.
- Graph inputs/outputs are free, but every internal tensor is expensive.
- Avoid large 14x14x10 intermediates unless absolutely necessary.
- Prefer scalar/small tensor logic, slicing, gather, reduce, equality, masks.
- Do not force a specific ONNX opset; use whatever gives lowest measured cost.

Expected strategy:
1. Read the 14x14 input, channel/grid format as used by existing NeuroGolf tasks.
2. Detect nonzero colored components/patterns.
3. Treat each candidate as a 3x3 bounding window around a non-background component.
4. Convert candidate pattern to a binary 3x3 mask:
   - cell = 1 where candidate color is present
   - cell = 0 elsewhere
5. Compare all candidate 3x3 binary masks.
6. Count how many times each mask occurs in the whole input, regardless of color.
7. Select the mask with the highest count.
8. Recover the winning color from one occurrence of that mask.
9. Output a 3x3 grid containing that color where mask=1, and 0 elsewhere.

For this task, exploit fixed size 14x14 and output 3x3:
- Enumerate only plausible top-left positions for 3x3 windows: rows 0..11, cols 0..11.
- For each window, identify whether it contains exactly one nonzero color and at least one nonzero cell.
- Encode 3x3 binary pattern as a 9-bit integer or equivalent compact representation.
- Count equal encodings.
- Pick argmax count.
- Reconstruct output from selected 9-bit mask and selected color.

Implementation requirements:
- Create a standalone Python script that generates `solution.onnx`.
- Hardcode paths, no argparse.
- Validate with onnx.checker and onnxruntime.
- Print NeuroGolf-relevant stats: initializer params count, inferred internal tensor memory if available, and output on provided examples.
- Try multiple graph variants and keep the one with smallest measured cost.

Aggressive optimization ideas:
- Since output is only 3x3, avoid producing a full labeled 14x14 map.
- Use fixed slices/gathers for 3x3 windows.
- Compare compact encoded masks instead of full 3x3 tensors where possible.
- If ONNX lacks convenient bit packing cheaply, use weighted sum with constants [1,2,4,8,16,32,64,128,256].
- Prefer uint/int/bool intermediates if accepted by ORT and shape inference.
- Avoid OneHot over 10 colors if it creates huge tensors.
- Avoid Conv unless it is clearly cheaper in params+memory.
- Constants count as params, so keep constants tiny.
- Final graph should output required NeuroGolf shape. If Kaggle wrapper requires [1,10,30,30], create only a tiny 3x3 internal result and pad/scatter at the very end, because graph output memory is free.

Deliverables:
- `build_task079.py`
- `solution.onnx`
- A small test runner that checks all train examples.
- A short report comparing attempted variants and final measured cost/score.

Be ruthless: if a generic connected-components implementation is expensive, replace it with fixed 

------------------------------------

tohle je task080 jdeo to ze nekde je nejaky patern a nahodne jeho streni bloky jinde na outputu je patern i na mistech kde byly bloky z jeho stredu. Dej mi prmpt pro vytvoreni onnx s nejvetsim moznym score

Použij tenhle prompt pro Codex/Cursor:

We need solve NeuroGolf ARC task080 by generating the smallest/highest-score ONNX possible.

Task:
- Input and output are always 26x26 ARC grids, one-hot encoded as [1,10,30,30] in the NeuroGolf wrapper.
- Real active area is top-left 26x26.
- Background is black/0.
- There is a repeated colored pattern somewhere in the grid.
- Elsewhere there are small partial markers equal to the CENTER block/center columns of that pattern.
- Output must paste the full original pattern at every marker location, aligned so the marker becomes the center of the pasted pattern.
- Existing full pattern remains unchanged.
- Everything outside active 26x26 should stay zero except required final padding.

Observed example:
- Full source pattern is a 6x6-ish multicolor block near top-left.
- Partial hints elsewhere are its middle vertical/central part.
- Output completes those hints into full copies of the source pattern.

Goal:
Create an ONNX model with maximum NeuroGolf score, not readability.

Optimization constraints:
- Minimize internal tensor memory + initializer scalar count.
- Do not optimize ONNX file size unless it reduces NeuroGolf cost.
- Avoid large [1,10,26,26] or [1,10,30,30] internal tensors.
- Work on compact spatial slices whenever possible.
- Final output must be [1,10,30,30], but graph inputs/outputs are free; only internal tensors and params count.
- Prefer Slice/Gather/Concat/Pad/Where/Equal/ArgMax/ReduceSum style ops over Conv.
- Avoid huge constants. Small coordinate/index constants are OK.
- Use dynamic detection only where it is cheaper than hardcoding.

Implementation strategy to try first:
1. Extract active area:
   - Slice input to [1,10,26,26].
   - Or avoid full active tensor if possible and work directly on needed coordinate slices.

2. Determine non-background occupancy:
   - Use channel/color info to identify nonzero cells.
   - The source pattern is the largest dense multicolor component.
   - The marker fragments are smaller components matching the source pattern’s center part.

3. Because the grid size is fixed 26x26, aggressively exploit this:
   - Build a Python generator that analyzes all train examples for task080.
   - Infer the exact source pattern bounding box size.
   - Infer the marker-center offset inside the source pattern.
   - Check if pattern size is constant across all examples.
   - Check if marker shape/center slice is constant.
   - If constant, hardcode only slice sizes/offsets, not colors.

4. ONNX graph logic:
   - Find the full pattern bbox from the largest nonzero connected component or from known top/left placement rules if train examples prove it.
   - Extract the source pattern as compact tensor P of shape [1,10,H,W].
   - Extract the center marker slice M from P using inferred offsets.
   - Search all possible marker locations in the 26x26 grid:
     - Compare local patch against M.
     - Candidate must match non-background cells of M.
     - Candidate should not be inside the original full pattern bbox.
   - For every candidate, paste P aligned by the center-marker offset.
   - Keep original input cells unless overwritten by a pasted full pattern.

5. Score-oriented alternatives:
   - If examples show the full source pattern is always at a fixed location, hardcode that slice. This is likely much cheaper than dynamic bbox detection.
   - If examples show marker locations are limited to a small set of grid-aligned placements, avoid full sliding-window search and only test those candidate positions.
   - If the pattern tile size is fixed, use fixed Slice + Pad for each possible placement, then combine with Where/Max.
   - If markers are always non-overlapping and color values are one-hot, combining pasted one-hot tensors with Max is usually cheap and valid.

6. Required deliverable:
   - Write a self-contained Python script `build_task080.py`.
   - It must build `solution.onnx`.
   - No argparse. Use hardcoded paths.
   - After saving, run ONNX checker and ORT inference on all available train examples.
   - Print exact match per example.
   - Print NeuroGolf-style cost estimate if available.
   - Try several graph variants automatically and keep the lowest-cost valid ONNX.

7. Do not assume opset 10. Use whatever opset gives the best score.
8. Prioritize measured NeuroGolf score over elegance.
9. If a general dynamic solution is expensive, create a specialized solution based on fixed task080 dimensions and train-set invariants.
10. The final model must solve hidden examples following the same task rule, not just memorize colors.

Nejspíš nejlepší score bude z varianty: hardcodnout bbox/velikost patternu + testovat jen možné marker pozice, ne dělat obecné connected-components v ONNX.

------------------------------------------------------------

tohle je task081 jde o to ze na vstupu jsou svetle modre nedokoncene ctverce a na vystupu jsou doplneny tmave modrou. Vstup je vzdy 7x7 Dej mi prompt pro optimalizovaneho onnx s nejlepsim moznym score.

Použij tenhle prompt pro Cursor/Codex:

You are optimizing NeuroGolf ARC task081.

Task:
Input/output grid is always 7x7.
Cells use ARC colors. Background is 0.
Input contains several incomplete 2x2 light-blue/cyan squares.
Each object is an L-shape made from exactly 3 cells of color 8 inside a 2x2 bounding box.
Output keeps all existing cyan cells and fills the missing fourth cell of every incomplete 2x2 square with dark blue color 1.

Goal:
Generate the smallest/highest-score ONNX model possible for this exact task.

Transformation:
For every 2x2 window:
- If exactly three cells are color 8 and one cell is 0, set the zero cell to 1.
- Otherwise leave cells unchanged.
All original color-8 cells remain 8.
No cropping/resize. Output shape should be 7x7, padded/wrapped only if the NeuroGolf harness requires fixed [1,10,30,30].

Implementation strategy:
Prefer a direct ONNX graph, not a neural network.
Use int/float tensor ops with minimal params.

Recommended graph:
1. Extract the active 7x7 plane from input.
2. Build cyan mask:
   m = Equal(x, 8)
3. For each missing-corner case, detect 2x2 L-shapes:

Top-left missing:
  x[r,c] == 0 and m[r,c+1] and m[r+1,c] and m[r+1,c+1]

Top-right missing:
  x[r,c+1] == 0 and m[r,c] and m[r+1,c] and m[r+1,c+1]

Bottom-left missing:
  x[r+1,c] == 0 and m[r,c] and m[r,c+1] and m[r+1,c+1]

Bottom-right missing:
  x[r+1,c+1] == 0 and m[r,c] and m[r,c+1] and m[r+1,c]

4. Scatter/compose a fill mask over the 7x7 grid where any missing corner was detected.
5. Output:
   y = Where(fill_mask, 1, x)

Optimization requirements:
- Avoid Conv if it creates large internal tensors.
- Avoid one-hot over 10 colors unless absolutely necessary.
- Avoid large 30x30 internal tensors. Work on 7x7 only and pad to required output shape at the final step.
- Use Slice/Equal/And/Or/Where/Concat/Pad or equivalent minimal ops.
- Constants should be scalar or very small tensors only.
- Do not include trainable weights.
- Minimize initializer count and scalar constants because NeuroGolf cost counts params.
- Measure score using the NeuroGolf scorer, not ONNX file size.

Important:
Test all four missing-corner orientations.
Test edge positions: 2x2 windows may start at rows/cols 0..5.
Multiple incomplete squares can exist independently.
Overlapping detections should be ORed; output color for inserted cells is always 1.

Deliverables:
- create task081.onnx
- include a small Python verification script that tests random 7x7 examples with multiple incomplete 2x2 cyan squares
- print NeuroGolf memory, params, cost, and score
- then try at least two variants:
  A) slice/logic/scatter version
  B) convolution/kernel-count version only if it has lower cost
Keep the model with the best measured score.

--------------------------------------

tohle je task082 jde o to ze na vstupu na prvnim radku jsou nejake bloky a ty jsou na vystupu utvoreny do 6 vysokeho paternu ktery se opakuje. Dej mi prompt pro vytvoreni optimalizovaneho onnx s nejlepsim moznym score

Tady je prompt pro Cursor/Codex:

We need solve NeuroGolf ARC task082 with an optimized ONNX model.

Task behavior:
- Input/output grid is always 6x10.
- Input contains nonzero colored cells only on the first row.
- Output keeps the first row unchanged.
- Rows 1..5 are generated from the first row by repeating a fixed 2-row vertical pattern:
  - even output rows 0,2,4: copy the original first row exactly.
  - odd output rows 1,3,5: transform the first row into expanded/duplicated positions:
    - for every nonzero cell at column c in the first row, place that same color in odd rows according to the learned horizontal phase pattern.
    - From example: red at c=1 creates red at c=0 and c=2 on odd rows; cyan at c=5 creates cyan at c=4 and c=6 on odd rows.
    - Equivalent: odd row = left-shift(first_row) OR right-shift(first_row), preserving each color.
- Background is 0.
- Final output shape must be [1,10,30,30] for NeuroGolf/Kaggle wrapper, but only the top-left 6x10 area is meaningful; rest should be zero padded.

Goal:
Create the smallest/cheapest valid ONNX possible. Optimize NeuroGolf score, not ONNX file size. Minimize internal tensor memory + params.

Implementation strategy:
- Input is one-hot [1,10,30,30].
- Extract only top-left 6x10 if needed, but actually only row 0 and cols 0..9 are needed.
- Work in compact tensors, preferably [1,10,1,10] or [1,10,6,10], never full 30x30 internally except final padded output.
- Get first row: x[:, :, 0:1, 0:10].
- Build even rows by copying first row to rows 0,2,4.
- Build odd row as:
  odd[:, :, :, 0:9] includes first_row shifted right? 
  odd at col c-1 = first_row at c, and odd at col c+1 = first_row at c.
  Use Slice + Pad/Concat, no parameters.
  left = pad first_row[:, :, :, 1:10] to columns 0..8
  right = pad first_row[:, :, :, 0:9] to columns 1..9
  odd = left + right
  Since ARC colors do not overlap in this task, Add is safe; otherwise use Where(nonzero).
- Stack rows:
  row0 = first_row
  row1 = odd
  row2 = first_row
  row3 = odd
  row4 = first_row
  row5 = odd
- Concat along height -> [1,10,6,10].
- Pad to [1,10,30,30] only at the final graph output.

Optimization requirements:
- Prefer ConstantOfShape / Slice / Concat / Pad with minimal constants.
- Avoid Expand/Tile to full [1,10,30,30].
- Avoid large initializers. Constants should be tiny int64 tensors for starts/ends/axes/pads only.
- No training. This is deterministic.
- After building ONNX, run checker, shape inference, ORT inference, and NeuroGolf scoring.
- Report memory, params, cost, score.
- Try variants and keep the best:
  1. Slice+Concat based shifting.
  2. Pad based shifting.
  3. If cheaper, prebuild final 6 rows using Concat only.
- Do not force a specific opset unless measurement proves it is best.

Expected test from screenshot:
Input first row has red at col 1 and cyan at col 5.
Output rows:
row 0 = original
row 1 = red at cols 0,2 and cyan at cols 4,6
row 2 = original
row 3 = same as row 1
row 4 = original
row 5 = same as row 1

--------------------------------------------

tohle je task083 jde o to ze na vstupu je patern z ktereho je utvoren vetsi patern tim ze se o 90 stupnu otaci. Vstup ma vzdy velikost 3x4. Dej mi prompt pro optimalizovane onnx s nejvetsim moznym score

Použij tento prompt pro Cursor/Codex:

We need solve NeuroGolf ARC task083 with a minimal-score ONNX generator.

Task rule:
- Input is always 3x4.
- Output is always 6x8.
- The 3x4 binary/color pattern is expanded into a 2x2 tiling of transformed copies:
  top-left     = original input
  top-right    = original rotated 180 degrees
  bottom-left  = original rotated 180 degrees
  bottom-right = original input

Equivalent:
Given X shape [1,10,30,30] one-hot ARC tensor, read only the active 3x4 region.
Let A = X[:, :, 0:3, 0:4].
Let R = reverse A in both spatial axes: A[:, :, ::-1, ::-1].
Output logical 6x8 is:
[[A, R],
 [R, A]]

Then pad/copy this 6x8 result into the required [1,10,30,30] output, with everything outside 6x8 set to zero/background.

Optimization goal:
- Maximize NeuroGolf score, not ONNX file size.
- Minimize internal tensor memory + params.
- Avoid large intermediate [1,10,30,30] tensors until final output if possible.
- Use Slice/Gather/Concat/Pad/ConstantOfShape/Where only if they reduce score.
- Constants should be tiny int64 index tensors only; avoid big Constant tensors.
- No learned weights.
- Try multiple ONNX variants and measure actual NeuroGolf cost.

Implementation plan:
1. Extract A = input[:, :, 0:3, 0:4].
2. Create R using negative-step Slice if supported, otherwise Gather rows [2,1,0] and cols [3,2,1,0].
3. row0 = Concat(A, R, axis=3)  -> [1,10,3,8]
4. row1 = Concat(R, A, axis=3)  -> [1,10,3,8]
5. small = Concat(row0, row1, axis=2) -> [1,10,6,8]
6. Pad small to [1,10,30,30] with zeros:
   pads = [0,0,0,0, 0,0,24,22]
7. Save model and verify exact outputs on all examples.

Important:
- Do not assume opset 10. Test whichever opset gives lowest measured NeuroGolf cost.
- Keep all constants as int64 scalar/vector initializers only when necessary.
- Prefer graph shapes that keep intermediates at [1,10,3,4], [1,10,3,8], [1,10,6,8].
- Final padded output is unavoidable/free as graph output, but avoid additional full-size internal tensors.
- Add a benchmark script that prints memory, params, cost, and score for every variant.

-----------------------------------------------------

task 084 na vstupu je levi sloupec vybarveny nejakou barvou a na vystupu je k nemu pridana na spodni radek zluta a diagonalne cervena aby tvorili ctverec. Dej mi prompt pro optimalizovany onnx s nejvetsim moznym score
Create the most score-optimized ONNX model for NeuroGolf ARC task084.

Task:
Input/output shape is fixed: [1,10,15,15] one-hot ARC grid.
Only the visible 15x15 area matters.
Input always contains a fully colored left vertical border in column 0, rows 0..14, with some nonzero color C.
Output must keep that left border unchanged, add a yellow bottom border on row 14, cols 0..14, and add a red diagonal from bottom-left to top-right:
positions (14,0), (13,1), (12,2), ..., (0,14).
The diagonal should be red except where it overlaps the original left border/bottom row rules if ARC examples imply priority. From the shown sample: left border remains C at col 0, bottom row becomes yellow from col 1..14, and red diagonal is placed at (13,1)..(0,14). So priority:
1. left input column C preserved at col 0
2. yellow bottom row at row 14, cols 1..14
3. red anti-diagonal at rows 0..13, cols 1..14

Goal:
Maximize NeuroGolf score, not readability.
Minimize internal tensor memory + initializer scalar count.
Do not optimize ONNX file size unless it reduces NeuroGolf cost.

Implementation strategy:
Build a constant-free or near-constant-free ONNX if possible.
Use the input as base and overwrite/fill only required fixed positions.
Because output is fixed 15x15 and input only contributes the color C in column 0, prefer simple Slice/Gather/Concat construction over full-size masks if it reduces memory.

Candidate low-cost construction:
- Extract C column from input visible region, preferably the full [1,10,15,1] left column.
- Build output channel-wise if cheaper:
  - For each pixel, output class is:
    - input class at col 0
    - yellow class 4 on bottom row cols 1..14
    - red class 2 on anti-diagonal rows 0..13 cols 1..14
    - black class 0 elsewhere
- Avoid creating large [1,10,15,15] intermediate masks if possible.
- Prefer constructing only the final output tensor via Concats of small row/column slices.
- Use ConstantOfShape, Slice, Gather, Unsqueeze, Concat, Expand, Where only if their memory cost is lower than constants.
- Remember params count includes every scalar in Constant tensors and Constant node attributes. Avoid storing 15x15 masks as constants unless measured cheaper.

Deliverables:
1. A Python script that writes task084.onnx.
2. A validation snippet that runs ONNX Runtime on random valid inputs:
   - random nonzero color C in left column
   - rest black
   - checks exact output against the rule above.
3. A small score-report helper printing inferred internal tensor shapes, initializer/Constant scalar count, and estimated NeuroGolf cost.
4. Try multiple graph variants and keep the smallest measured NeuroGolf cost:
   - variant A: Slice/Concat rows
   - variant B: one-hot constant index grid + Equal/Where
   - variant C: direct ScatterND/ScatterElements update if supported and cheaper
   - variant D: build compact [15,15] class-index grid then OneHot only at the end
5. Do not force a specific opset. Test opsets and keep the best score-compatible one.

Important:
The output must be [1,10,15,15] float32 one-hot.
Do not pad to 30x30 unless the task evaluator explicitly requires it.
Keep graph valid under ONNX shape inference and ONNX Runtime.

--------------------------------------------

task085 na vstupu jsou obdelniky 3 bloky vysoke na vystupu maji obdelniky prostreni radek smazany kazdy dryhy blok. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt pro Cursor/Codex:

We need solve NeuroGolf ARC task085 with a maximally score-optimized ONNX model.

Task rule:
Input and output shape are identical, examples include 8x30 and 11x20.
The grid contains one or more solid horizontal rectangles, always exactly 3 rows high, arbitrary width, arbitrary color, on black background.
For every such 3-row-high rectangle:
- keep the top row unchanged
- keep the bottom row unchanged
- modify only the middle row
- in the middle row, erase every second colored cell to black, producing an alternating pattern
- the alternating phase is local to the rectangle span: keep column 0 of the rectangle, erase column 1, keep column 2, erase column 3, etc.
All non-rectangle/background cells stay unchanged.

Goal:
Create the smallest/cheapest ONNX graph for NeuroGolf score, not a generic neural net.

Important optimization constraints:
- Graph input/output may be [1,10,30,30], but internal tensors must be minimized.
- Large internal [1,10,30,30] tensors are expensive. Avoid them except final pad/output if required.
- Use integer/boolean masks where possible.
- Prefer Slice/Gather/Concat/Where/Equal/Cast over Conv or heavy ops.
- Do not optimize ONNX file size unless it reduces params or internal tensor memory.
- Params count initializers/constants, so keep constants tiny.
- Any opset is allowed if it improves score.

Implementation idea:
1. Slice active HxW area from channel 0.
2. Detect colored cells: x != 0.
3. Detect middle row of a 3-high horizontal rectangle:
   mid = colored[r-1,c] & colored[r,c] & colored[r+1,c]
   This works because rectangles are 3 rows high.
4. Build alternating erase mask by column parity relative to the start of each horizontal run.
   Since each rectangle row is continuous:
   - find run start per row: colored[r,c] & not colored[r,c-1]
   - propagate cumulative parity across columns inside each run, or cheaper if width/positions are fixed per task examples.
   - erase cells where mid == true and local_index % 2 == 1.
5. output = where(erase_mask, 0, input)
6. Pad/cast back to required NeuroGolf output shape.

Search for cheaper variants:
- If task inputs have fixed dimensions per test split, generate specialized graphs per detected input H/W.
- If rectangles always begin at fixed known parity positions, use global column parity instead of local run parity. Validate against examples; if all rectangle starts share same parity, this is much cheaper.
- Try both:
  A) global parity erase: erase mid rows at odd/even absolute columns
  B) local run parity erase
  Keep the model with lower NeuroGolf cost that still passes all train examples.

Deliverables:
- Python script that builds model.onnx directly using onnx.helper.
- Validation script comparing ONNXRuntime output against all train examples.
- Print NeuroGolf-relevant estimates: params count, inferred internal tensor shapes, memory estimate.
- Iteratively minimize graph cost and keep only the best valid ONNX.

Do not create a learned model. This is a deterministic ARC transform.

Největší šance na top score: nejdřív zkus globální paritu sloupců, protože podle screenshotů mazání sedí na absolutní každý druhý sloupec. Pokud projde train set, je levnější než počítat začátek každého obdélníku.

---------------------------------------------

tohle je task086 na vstupu je ctverec se stredem jine barvy na vystupu ma ctverec barvu otocenou se stredem a pribidou mo kostky po sranach podle velikosti stredu. Dej mi prompt pro onnx s nejlepsim moznym score

Create the most score-optimized ONNX solver for NeuroGolf ARC task086.

Task:
Input and output are always 10x10 grids, one-hot ARC format [1,10,10,10].
The input contains one centered square/ring object on black background:
- outer square/ring color A
- inner center block color B
- object is axis-aligned
- center block can be 1x1 or 2x2
- outer square is one-cell thick around the center block

Output rule:
1. Swap the two object colors inside the original square:
   - cells that were outer color A become B
   - cells that were center color B become A
2. Add four side “arms” around the square, using the original outer color A.
3. Arm thickness/length is determined by the center size:
   - if center is 1x1, add 1-cell-thick arms: one cell outward from each side position around the square, forming a plus-like expansion.
   - if center is 2x2, add 2-cell-thick rectangular arms on left/right/top/bottom, matching the 2-wide center span.
4. Background stays black.
5. Preserve 10x10 output size.

Implement directly as a minimal ONNX graph optimized for NeuroGolf score:
- Minimize internal tensor memory and initializer params.
- Do not optimize for ONNX file size; optimize only `cost = internal_memory + params`.
- Graph input/output tensors are free, but internal tensors are expensive.
- Avoid large [1,10,10,10] intermediates whenever possible.
- Work mostly on compact spatial slices / masks / coordinate vectors.
- Pad/scatter to the final required [1,10,10,10] only at the end.
- Prefer bool/int64 masks where possible if cheaper, cast only when needed.
- Avoid Conv, large constants, large one-hot tensors, and unnecessary full-grid temporaries.
- Use dynamic extraction of the nonzero object bbox:
  - detect non-black cells from channels 1..9
  - find min/max row/col of the object
  - infer center bbox as cells inside object with color B
  - infer color A and B from the input, not hardcoded.
- Build output from compact coordinate masks:
  - original object bbox with colors swapped
  - four arm regions with color A
- Convert final class grid to one-hot [1,10,10,10].

Return:
- A complete self-contained Python script that builds and saves `task086.onnx`.
- Include a tiny local verifier using the shown examples.
- Print NeuroGolf-relevant shape/memory diagnostics after ONNX shape inference.
- Try several implementation variants if useful and keep the lowest-cost valid model.

-------------------------------------------

tohle je task 087 na vstupu je vzdy 3x3 vystup je vstup flipnuty zvrchu dolu a zleva doprava. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt pro Cursor/Codex:

We need solve NeuroGolf ARC task087 with an extremely compact ONNX model.

Task:
- Input is always a 3x3 ARC grid.
- Output is the same 3x3 grid flipped both vertically and horizontally.
- Equivalent to 180-degree rotation:
  out[r, c] = input[2-r, 2-c]

Goal:
Create the lowest-cost valid ONNX model for NeuroGolf.
Optimize score, not readability or ONNX file size.

Important optimization constraints:
- Input/output tensors are free, but internal tensors and initializers are expensive.
- Avoid constants/initializers if possible.
- Avoid OneHot, Cast-heavy pipelines, big masks, Expand, Tile, or any internal [1,10,30,30] tensors.
- Since shape is fixed 3x3 -> 3x3, prefer the smallest graph that directly reverses both spatial axes.
- Use pure tensor slicing/gather/transpose-style ops only if they minimize params + memory.
- Measure actual NeuroGolf score after every variant.

Try variants:
1. Best candidate: use Slice on H and W axes with negative step to reverse both dimensions directly.
   If input shape is [1,10,3,3], output = input[:, :, ::-1, ::-1].
2. Alternative: two Gather ops with indices [2,1,0] on height and width.
3. Alternative: one Flatten/Gather/Reshape if it scores lower, with gather order [8,7,6,5,4,3,2,1,0] per channel/grid.
4. If input is [3,3] or [1,3,3], adapt axes accordingly, but keep graph minimal.

Build several ONNX candidates and run:
- shape inference
- ONNX Runtime validation
- NeuroGolf scoring
- compare memory, params, cost, score

Expected behavior example:
Input:
a b c
d e f
g h i

Output:
i h g
f e d
c b a

Return:
- best ONNX file
- exact measured score/cost/memory/params
- short explanation why this variant is optimal

--------------------------------------------------------


tohle je task 088 jde o to ze na vstupu je patern v jedne barve ktery je ohraniceny bloky jine barvy v rozich vystup je pak patern v barvne kterou je ohraniceny. Dej mi prompt pro vytvoerni perfektniho onnx s nejlepsim moznym score
Přemýšlení několik sekund

Zkopíruj do Codex/Cursor:

We need create the smallest/highest-score ONNX model for NeuroGolf ARC task088.

Task rule:
Input contains:
- background 0
- one “pattern color” object/pattern, arbitrary color
- several corner/marker cells of another color around the pattern
Output is the extracted pattern, but recolored to the marker/border color.
The output crop is the minimal bounding box of the pattern-color cells only.
All non-pattern cells inside that crop become 0.
All pattern-color cells become the marker color.

Examples:
7x7 -> 3x3:
red pattern inside, yellow corner markers around it.
Output is the red pattern cropped to bbox, recolored yellow.

22x9 -> 10x4:
blue pattern inside, red marker blocks around it.
Output is blue pattern cropped to bbox, recolored red.

Important:
- Marker color is the nonzero color that is NOT the main pattern color.
- Pattern color is the nonzero color with the actual connected/large shape, usually more cells.
- Need output dynamic crop size based on bbox of pattern color.
- Do NOT include marker blocks in output bbox.
- Output tensor can be ARC required padded output if NeuroGolf harness requires fixed [1,10,30,30], but internally keep tensors tiny.
- Optimize score = minimize params + internal tensor memory.
- Prefer constants, Slice/Gather/Where/Equal/Reduce ops.
- Avoid one-hot [10,H,W] if possible.
- Use uint8/int64/bool where legal to reduce memory; avoid float.
- No unnecessary large intermediates.
- ONNX may use any opset if it improves score.

Implementation strategy:
1. Input shape is [1,10,H,W] one-hot ARC format or whatever the local NeuroGolf template uses. Inspect existing tasks in repo.
2. Convert to compact grid [H,W] with ArgMax over channel if needed.
3. Find nonzero colors and counts.
4. Marker color = nonzero color with smaller count / color not forming the bbox pattern.
5. Pattern color = other nonzero color.
6. mask = grid == pattern_color.
7. Compute rows/cols where mask has any true.
8. Find min_row, max_row, min_col, max_col.
9. Slice mask bbox.
10. Output crop = Where(mask_crop, marker_color, 0).
11. Pad/write into required output tensor shape only at the very end.

Please generate a Python script that builds `task088.onnx`, runs ONNX checker, and locally tests on hardcoded examples. Then run the NeuroGolf scorer and report memory, params, cost, score. Iterate aggressively for lower score cost.

------------------------------------------------------------

tohle je task089 na vstupu jsou jeden nebo dva peterny tvorene z dvou barev jedna barva se tam vyskytuje pouze ve jedom bloku blok tehle barvy se pote nachazi na dalsich mistech na vystup se tento pater nakopiruje na dana mista. vstup je vzdy 13x13. Dej mi prompt pro perfektni onnx a nejlepsim moznym score.
You are optimizing NeuroGolf ARC task089. Build the smallest/highest-score ONNX solver possible. Do not optimize for file size; optimize NeuroGolf score = low params + low internal tensor memory.

Task rule:
Input/output are always 13x13.
The grid contains one or two complete 2-color patterns. Each pattern has:
- a dominant color occupying several cells,
- a marker/anchor color that appears exactly once inside that pattern.
The same marker color also appears elsewhere as isolated single cells.
For every isolated marker cell, copy the full 2-color pattern so that the marker cell in the copied pattern lands exactly on that marker position.
Keep the original complete pattern(s). Background stays 0.

Example behavior:
If a blue/green L-like pattern has a single green cell inside it, and another green singleton appears elsewhere, paste the whole blue/green pattern around that singleton using the green cell as anchor.
If an orange/red pattern has a single red cell inside it, and another red singleton appears elsewhere, paste that orange/red pattern around the red singleton using the red cell as anchor.
There may be one or two such marker colors/patterns in the 13x13 input.

Goal:
Generate an ONNX model with minimal NeuroGolf cost. Use fixed 13x13 assumptions aggressively. Avoid large float intermediate tensors. Prefer uint8/bool/int64 masks, Slice, Equal, Where, Or, Add/Mul only if cheaper. Avoid Conv and big one-hot expansions unless absolutely necessary.

Implementation strategy:
1. Input shape is [1,10,13,13] one-hot ARC format.
2. Convert/find color plane occupancy for colors 1..9; background is 0.
3. For each nonzero color c:
   - count total cells of color c over 13x13.
   - marker colors are colors with count > 1 but with isolated singletons plus one occurrence inside a larger 2-color component, OR in this task simpler: the rare color of each pattern is the color that appears as exactly the anchor count positions, while its partner color appears clustered around one of those anchors.
4. Detect marker colors by testing candidate anchor cells of color c and looking in a small local window around each occurrence for a neighboring nonzero different color.
   - The occurrence with nearby partner cells is the source anchor.
   - Other occurrences of the same color with no nearby partner cells are target anchors.
5. Extract the source pattern as a compact local window around the source anchor.
   - Search only small offsets; from examples pattern fits inside about 3x3/4x4.
   - Encode relative offsets from anchor for all cells in the source connected component containing source anchor and partner color.
   - Copy those relative offsets to every target anchor of the same marker color.
6. Because grid is fixed 13x13, implement copy by shifted masks:
   For each possible relative offset (dy, dx) in a bounded window around source anchor:
   - source_has_cell_at_offset = whether source component has a nonzero cell at anchor+(dy,dx)
   - source_color_at_offset = color at that source relative cell
   - for every anchor mask of this marker color, shift it by (dy,dx)
   - paint shifted target positions with source_color_at_offset.
7. Merge copied patterns with original input. Preserve original nonzero cells.

Important optimization constraints:
- Do not materialize full [1,10,13,13] intermediate repeatedly. Work mostly with [13,13] masks per color or compact scalar counts.
- Final output must be [1,10,13,13] one-hot.
- Internal tensors larger than final output are bad. Large Pad/Concat chains are acceptable only if they reduce total memory versus simpler but huge tensors.
- Constants count as params, so avoid lookup tables with many scalar constants. Use graph structure and tiny scalar constants where possible.
- Fixed-size Slice/Pad shifts are likely cheaper than dynamic Gather/Scatter if they avoid big index tensors.
- Avoid NonZero/TopK if they create large int64 tensors unless measured cheaper.

Build plan:
A. First implement a correct reference Python function for the 13x13 grid decoded to int colors:
   - connected components over nonzero cells, 4-neighbor or 8-neighbor; use 8-neighbor if needed by examples.
   - identify components with exactly two colors and one color count == 1 inside component.
   - that single-count color is marker color; its coordinate is source anchor.
   - all other cells of the same marker color outside the component are target anchors.
   - paste the whole component relative to source anchor onto every target anchor.
   - repeat for all valid source components.
B. Then translate this into ONNX with fixed-shape operations. If full connected-component logic is expensive, exploit task constraints:
   - source pattern is the occurrence of marker color that has adjacent nonzero partner cells.
   - target anchors are same-color cells not adjacent to partner cells.
   - relative stencil window around source anchor is small and fixed.
C. Generate several candidate ONNXs:
   1. simplest correct shifted-mask implementation,
   2. smaller-memory variant using per-color loops unrolled,
   3. variant using decoded [13,13] int grid then final one-hot only at the end.
D. Run official NeuroGolf validation/scoring for each candidate. Report:
   - correctness on train/test examples,
   - memory,
   - params,
   - cost,
   - score.
E. Keep only the best-scoring valid ONNX.

Do not assume opset 10. Try any opset that improves score. The winning model is the one with lowest measured NeuroGolf cost, not the prettiest graph.

Deliverables:
- `solution_task089.onnx`
- build script
- scorer output
- short explanation of the chosen graph and why it has the best measured score.

-----------------------------------------------

tohle je task090. Dej mi prompt na vytvoreni perfektne optimalizovaneho onnx s nejlepsim moznym score
Přemýšlení po dobu 10s
Create the smallest/highest-score ONNX for NeuroGolf task090.

Task:
Input/output shape is always Hx20, where H is 2 or 4.
Only active rows contain black background 0, blue 1, and output adds purple 6.

Transformation:
For every row independently:
1. Find the first blue run from the left.
2. Find the next blue cell/run to its right.
3. Fill all black cells strictly between the end of the first blue run and the start of the next blue run with purple 6.
4. Keep all original blue cells unchanged.
5. Keep all other cells black.

Examples:
[1,1,0,0,0,0,0,0,0,1,...] -> [1,1,6,6,6,6,6,6,0,1,...]
[0,1,0,0,0,0,0,1,1,1,...] -> [0,1,6,6,6,6,6,1,1,1,...]

Optimization goal:
Minimize NeuroGolf score cost = params + internal tensor memory.
Do not optimize ONNX file size unless it reduces params or internal tensors.

Implementation constraints:
- Prefer pure tensor ops, no large one-hot expansion.
- Input/output should stay [1,10,30,30] only if required by harness, but internally crop/use only Hx20 region.
- Use integer/bool masks, avoid float tensors.
- Avoid creating large [1,10,30,30] intermediates.
- Use constants only when necessary.
- Compute row-wise over width=20 using compact tensors.

Suggested ONNX logic:
- Extract active grid region: first H rows and 20 columns from channel 0.
- blue = (x == 1)
- For each row, compute cumulative count of blue cells from left.
- Identify cells after the first blue run:
  after_first = cumulative_blue > size_of_first_run or equivalently “seen at least one blue and currently past first contiguous run”.
- Simpler robust method:
  first_blue_col = ArgMax(blue along width)
  has_blue = ReduceMax(blue)
  after_first_start mask: col > first_blue_col
  next_blue_col = first blue col where col > first_blue_col and blue==1
  fill_mask = col > first_blue_col AND col < next_blue_col AND x==0
  output = Where(fill_mask, 6, x)
- If the first blue object can be a run, use end of first run:
  first_run_mask = blue AND cumulative_blue == col-relative contiguous prefix after first_blue_col
  first_run_end = last col of first contiguous blue run
  fill between first_run_end and next blue after it.

Deliver:
- A Python script that builds the ONNX.
- Then run NeuroGolf validation/profiling and print memory, params, cost, score.
- Iterate to reduce internal tensors and constants.
- Do not assume a fixed opset; choose whichever gives the lowest measured cost.

-----------------------------------------

tohle je task091 na vstupu jsou svisle dve sede cary na vystupu je vnitrech techle car plus jeden blok nad nimi a pod nimi. Dej mi prompt pro onnx s nejlepsim moznym score
Přemýšlení několik sekund

Tady je prompt pro Cursor/Codex:

We need solve NeuroGolf ARC task091 by generating a maximally score-optimized ONNX model.

Task:
Input grid is always HxW with H/W varying, examples include 11x13 -> output 6x5 and 9x11 -> output 5x7.
Input contains exactly two vertical gray bars (color 5). These bars define the left/right boundary.
Output is the rectangular region between these two gray bars, including:
- the gray bar columns themselves,
- all columns between them,
- one row above the top of the gray bars,
- one row below the bottom of the gray bars.

Then crop that rectangle from the input.

More formally:
1. Find all pixels with value 5.
2. Their columns are exactly two distinct columns: c0 and c1.
3. Their rows form a vertical interval from r0 to r1.
4. Output = input[r0-1 : r1+2, c0 : c1+1].
5. Preserve all colors exactly: cyan objects inside/around the crop remain, gray bars remain, black background remains.

Optimization goal:
Generate the smallest/highest-score ONNX possible for NeuroGolf.
Do NOT optimize file size unless it reduces NeuroGolf cost.
Minimize:
- internal tensor memory after shape inference/profiling,
- number of scalar params/constants.
Graph input/output tensors are free; internal tensors are expensive.
Avoid building full one-hot tensors or large broadcast masks unless absolutely necessary.

Implementation strategy:
- Since grid sizes are fixed per model/task family only to small known max, prefer direct tensor ops over general dynamic algorithms.
- Use ONNX ops that compute:
  - gray mask = Equal(input, 5)
  - reduce over rows/cols to find gray columns and gray rows.
  - derive min/max row/col indices for gray pixels.
  - Slice input using dynamic starts/ends.
- Output shape must be dynamic-compatible with the task cases, e.g. 6x5 or 5x7 depending on input.
- If dynamic Slice causes too much memory/params, test an alternative specialized graph using small constants for index vectors and ArgMax/NonZero/Gather.
- Avoid padding to 30x30 unless Kaggle wrapper absolutely requires it; if required, keep crop internal small and pad only as final output.

Deliverables:
1. A Python script that creates `task091.onnx`.
2. A local verifier that runs ONNX Runtime on provided train examples and checks exact output.
3. Print NeuroGolf-relevant estimates: initializer scalar count, inferred intermediate tensor shapes, and approximate internal memory.
4. Try at least 3 graph variants and keep the lowest-cost valid model:
   - dynamic Slice using ReduceMax/ArgMax,
   - NonZero/Gather min/max,
   - hardcoded-rank compact mask/index method.

Important:
Correctness first, then score.
Do not assume a specific opset unless measured better.
Use the smallest dtype possible for constants.
Avoid unnecessary Casts, Reshapes, Expand, Tile, or one-hot tensors.

--------------------------------------

tohle je task092 na vstupu jsou dva bloky ruznych barev ktere jsou vertikalne nebo horizonatlne spojeny danou barvou prekrivaji li se tak zustane barva ktera vede vertikalne. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Prompt pro Cursor/Codex:

Create the most score-optimized ONNX solver for NeuroGolf ARC task092.

Task rule:
- Input and output are always 30x20 -> 30x20.
- Background is 0.
- Input contains pairs of same-colored cells.
- For each color, there are exactly two cells aligned either vertically or horizontally.
- Output draws a solid straight segment connecting the two cells of that color:
  - if same row: fill all cells between min_col and max_col inclusive with that color
  - if same column: fill all cells between min_row and max_row inclusive with that color
- If a horizontal and vertical segment overlap, the vertical segment wins and remains visible at the crossing.

Optimization goal:
- Minimize NeuroGolf cost = internal tensor memory + initializer params.
- Do not optimize ONNX file size unless it reduces memory/params.
- Inputs/outputs are free, internal tensors are expensive.
- Prefer int64/bool logic only where needed; avoid large float tensors.
- Avoid full one-hot color expansion if possible.
- Keep graph compact and deterministic.
- Output shape must be exactly [1,10,30,20] one-hot/color-channel format required by the evaluator.

Implementation strategy:
1. Extract the active 30x20 grid from input.
2. Find nonzero cells and their colors.
3. For each color 1..9:
   - build mask of positions equal to color
   - detect whether the two cells share row or column
   - compute min/max row and col
   - generate either vertical or horizontal line mask
4. Compose output with priority:
   - first draw horizontal lines
   - then overwrite with vertical lines
5. Convert final colored grid to required one-hot [1,10,30,20].
6. Keep all intermediate tensors as small as possible; avoid materializing [10,30,20] repeatedly.
7. Measure actual NeuroGolf cost after shape inference/profiling and iterate.

Important:
- Vertical lines must override horizontal lines at intersections.
- Do not hardcode colors from examples; work for arbitrary colors 1..9.
- Do not assume fixed positions.
- The best solution is likely a small generic graph using coordinate masks and color equality, not a lookup table.
- After creating the ONNX, run the official validator/profiler and print memory, params, cost, and score.

--------------------------------------------

tohle je task093 na vstupu jsou bud vertikalne nebo horiznotalne sede casry a k nim bloky jine barvy ktere josu prisunuty k sedim caram a zabarvene do seda na vystu. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create an ultra-minimal ONNX solver for ARC/NeuroGolf task093.

Task rule:
Input/output are always 14x14 one-hot ARC tensors [1,10,30,30], but only the top-left 14x14 region matters.

The input contains:
- one gray horizontal or vertical thick line / band, color 5
- several colored marker pixels, all same non-gray color
- output keeps the gray band unchanged
- every colored marker is projected straight toward the gray band, then recolored gray
- projection direction is perpendicular to the band:
  - if gray band is horizontal, each marker becomes gray at the same column, directly adjacent to the band edge
  - markers above the band are placed one row above the band
  - markers below the band are placed one row below the band
  - if gray band is vertical, each marker becomes gray at the same row, directly adjacent to the band edge
  - markers left of the band are placed one column left of the band
  - markers right of the band are placed one column right of the band
- original colored marker pixels disappear
- everything else is black
- final output must be padded back to [1,10,30,30].

Optimization goal:
Minimize NeuroGolf cost = internal tensor memory + initializer params. Do not optimize ONNX file size unless it also reduces memory/params.

Implementation strategy:
Use ONNX ops only, no Python runtime logic. Prefer constants/slices/gather/scatter/where over large intermediate tensors.

Recommended construction:
1. Slice top-left 14x14 from class channels.
2. Extract gray mask channel 5: G = x[:,5,0:14,0:14].
3. Extract non-background/non-gray marker mask:
   M = any over channels except 0 and 5, shape 14x14.
4. Determine orientation cheaply:
   - row_count = ReduceSum(G, axis=cols)
   - col_count = ReduceSum(G, axis=rows)
   - horizontal if max(row_count) > max(col_count), else vertical.
5. For horizontal:
   - band rows are rows where row_count > 0.
   - top_edge = min gray row.
   - bottom_edge = max gray row.
   - markers above: M rows < top_edge -> place at row top_edge-1, same col.
   - markers below: M rows > bottom_edge -> place at row bottom_edge+1, same col.
6. For vertical:
   - band cols are cols where col_count > 0.
   - left_edge = min gray col.
   - right_edge = max gray col.
   - markers left: M cols < left_edge -> place at col left_edge-1, same row.
   - markers right: M cols > right_edge -> place at col right_edge+1, same row.
7. Output gray mask = original G OR projected_marker_positions.
8. Build one-hot output:
   - channel 5 = gray output mask
   - all other channels = 0
   - pad spatially to 30x30.

Important optimization constraints:
- Avoid creating [1,10,14,14] intermediates except final output if possible.
- Work mostly with [14,14], [14], and scalar tensors.
- Avoid large Tile/Expand.
- Avoid float operations if int/bool works.
- Use bool masks where possible.
- Minimize Constant tensors. Small coordinate constants 0..13 are acceptable.
- Prefer ArgMax on row/column presence vectors to find edges:
  - top_edge = ArgMax(row_has_gray)
  - bottom_edge = 13 - ArgMax(Reverse(row_has_gray))
  - left_edge/right_edge analogous.
- Projection can be implemented with equality against row/col coordinate vectors:
  - horizontal projected mask =
    ((row == top_edge-1) AND any_marker_above_by_col) OR
    ((row == bottom_edge+1) AND any_marker_below_by_col)
  - vertical projected mask =
    ((col == left_edge-1) AND any_marker_left_by_row) OR
    ((col == right_edge+1) AND any_marker_right_by_row)
- Select horizontal/vertical result with Where(horizontal_condition, h_result, v_result).

Deliverables:
- A script that writes task093.onnx.
- Include a small local verifier with manually constructed examples matching the rule.
- Print NeuroGolf-relevant stats: initializer scalar count, inferred internal tensor shapes, and estimated internal memory.
- Try multiple graph variants if useful and keep the lowest cost.

Do not assume opset 10. Use whichever opset gives the smallest measured NeuroGolf score.

-----------------------------------------

tohle je task094 na vstupu jsou tmave modre ctverce na svetle modrem pozadi je pres ne vepsano + na vystup. vstup je vzdy 15x15.  Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Přemýšlení několik sekund
Task094 NeuroGolf / ARC-AGI ONNX optimalization prompt

Vytvoř co nejmenší a nejlevnější ONNX model pro ARC task094.

Úloha:
- Input i output mají vždy přesně tvar 15x15.
- Pozadí je světle modrá / cyan barva.
- Na vstupu je jeden tmavě modrý čtvercový/rámový objekt na cyan pozadí.
- Output je stejný jako input, ale přes střed tmavě modrého objektu je doplněné magenta `+`.
- Magenta plus má:
  - horizontální čáru přes celý řádek středu objektu, od levého okraje gridu po pravý okraj gridu,
  - vertikální čáru přes celý sloupec středu objektu, od tohoto středového řádku směrem dolů až po spodní okraj gridu.
- Tmavě modrý objekt zůstává zachovaný, ale magenta `+` ho přepíše tam, kde se kříží.

Implementační cíl:
- Maximalizuj NeuroGolf score, tedy minimalizuj `memory + params`.
- Neoptimalizuj velikost ONNX souboru, optimalizuj skutečný NeuroGolf cost.
- Input/output tensor `[1,10,30,30]` je Kaggle wrapper; interně pracuj jen s levným výřezem 15x15 a výsledek až na konci padni do požadovaného tvaru.
- Nepoužívej Conv, velké one-hot mezitensory ani zbytečné Cast/Where řetězce.
- Preferuj Slice, ReduceMax/ReduceSum, ArgMax/NonZero/Gather, Equal, Less/Greater, And/Or, Where, Expand, Range, Unsqueeze.
- Použij dtype co nejlevnější pro masky, ideálně bool/uint8, a až výstup drž jako int64/float podle požadavků evaluátoru.

Detekce:
1. Vezmi top-left 15x15 z inputu.
2. Najdi všechny buňky s tmavě modrou barvou. Barvu nehardcoduj, pokud to jde levně:
   - pozadí je dominantní cyan,
   - objekt je nedominantní nenulová barva odlišná od pozadí,
   - v trénovacích příkladech je tmavě modrá typicky hodnota 1.
   Pokud hardcoding barvy 1 výrazně sníží cost a task ji vždy používá, preferuj hardcoding.
3. Spočítej bounding box tmavě modrého objektu:
   - `rmin, rmax, cmin, cmax`.
4. Střed plusu:
   - `center_row = (rmin + rmax) // 2`
   - `center_col = (cmin + cmax) // 2`
5. Vytvoř masku magenta plus:
   - `plus_h = row == center_row` pro všechny sloupce 0..14
   - `plus_v = (col == center_col) AND (row >= center_row)` pro řádky center_row..14
   - `plus = plus_h OR plus_v`
6. Output15:
   - `where(plus, 6, input15)`
   - magenta barva je hodnota `6`.
7. Finální výstup:
   - padni `output15` do `[1,10,30,30]` podle NeuroGolf/Kaggle formátu, bez vytváření velkých interních tensorů dřív než na posledním kroku.

Důležité:
- Všechny pomocné masky drž maximálně ve velikosti 15x15 nebo ještě lépe 15 řádků / 15 sloupců.
- Nepočítej celý `[1,10,30,30]` interně před finálním padem.
- Pokud lze řádkovou/sloupcovou masku vytvořit jako broadcast z vektorů délky 15, udělej to.
- Cílem je perfektní generalizace pro libovolnou pozici tmavě modrého objektu ve 15x15 gridu.
- Po vytvoření modelu spusť shape inference + ORT profiling a vypiš NeuroGolf memory, params, cost a score.
- Pokud existují dvě varianty, vyber tu s nižším `memory + params`, ne tu s kratším kódem.

----------------------------------------

tohle je taks095, vstup je vzdy 9x9. na vstupuju jsou sede bloky kterou jsou obklopeny modrimy bloky na vystupu. Dej mi prompt pro onnx s nejlepsim moznym score

Prompt pro Cursor/Codex:

We need solve NeuroGolf ARC task095 with a minimal-cost ONNX model.

Task:
- Input is always 9x9, one-hot ARC tensor likely [1,10,9,9].
- Background is black/color 0.
- Input contains several isolated gray cells, color 5.
- Output keeps each gray center and surrounds every gray cell with a full 3x3 blue square, color 1.
- In other words: for every input cell == 5, set all cells within Chebyshev distance <= 1 to blue, except the original center remains gray.
- Multiple 3x3 neighborhoods may exist; if they overlap, gray centers must remain gray.
- Output size remains 9x9, no 30x30 padding unless evaluator requires it.

Optimization goal:
- Maximize NeuroGolf score: minimize internal tensor memory + initializer params.
- Do not optimize ONNX file size unless it reduces params/memory.
- Avoid large one-hot intermediates if possible.
- Prefer operations on a single channel/mask.

Implementation idea:
1. Extract gray channel 5 from input: mask = X[:,5:6,:,:].
2. Build 3x3 dilation of gray mask.
   - Best option: MaxPool with kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1].
   - This creates a 1-channel 9x9 mask of all cells around gray centers.
3. Build output one-hot:
   - blue channel = dilated_mask AND NOT gray_mask
   - gray channel = gray_mask
   - all other channels = 0
4. Return [1,10,9,9] one-hot float/bool-like tensor.

Try variants and measure score:
A) MaxPool dilation variant.
B) Conv with fixed 3x3 ones only if params/memory score is better; likely worse because 9 params.
C) If ReduceMax/Pad/Slice shifts are cheaper than MaxPool in this scoring setup, test 9 shifted masks OR’ed together, but likely more intermediate memory.

Important:
- Preserve gray center as color 5.
- Surrounding 8 neighbors are blue color 1.
- Border gray cells should only fill valid in-grid neighbors.
- Use constants as small scalar/list attributes where possible.
- Run ONNX shape inference and ORT profiling, then report memory, params, cost, and score.
- Keep the graph simple and deterministic.

-----------------------------------------------------------

Tohle je task096 na vstupu jsou ctverce ruzne velikosti ktere jsou do sebe vlozeny na vystup. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Create the smallest/highest-score ONNX solver for NeuroGolf task096.

Task:
- Input is always 13x17.
- Output is always 7x7.
- Grid contains nested square/rectangle frames/patterns of different colors.
- The output is a compressed 7x7 representation of the nested squares.
- Preserve the colors and nesting order:
  - outermost border color becomes 7x7 border
  - next inner frame becomes 5x5 border
  - next inner frame becomes 3x3 border
  - center cell is the innermost color
- Ignore background/empty spacing and irrelevant repeated filler structure.
- The transformation is essentially: detect the visible nested square colors from outside to inside, then emit canonical 7x7 nested-square template.

Optimization goal:
- Maximize NeuroGolf score, not readability.
- Minimize internal tensor memory + initializer params.
- Inputs/outputs are free, but internal tensors are expensive.
- Avoid large one-hot tensors, avoid full [1,10,13,17] intermediates if possible.
- Prefer scalar/small-vector ops, Gather/Slice/Reduce/Where/Equal over Conv or large constants.
- Constants should be tiny int64/int32 index arrays or scalar colors only.
- Final output must be [1,10,30,30] if the evaluator requires padded ARC format, but keep the real constructed tensor at 7x7 and pad only at the very end.

Implementation strategy:
1. Extract the 13x17 class grid from channel 0 / argmax input format as appropriate for the existing NeuroGolf harness.
2. Identify the 4 nested colors robustly from the input:
   - outer frame color: dominant non-background/yellow-brown frame color in the long horizontal bands.
   - second frame: color appearing in the next nested square positions.
   - third frame: color appearing in the inner square positions.
   - center color: innermost cell color.
3. Because input size is fixed, prefer fixed sampled coordinates over generic connected-components:
   - Sample representative coordinates for each nested level from training-consistent positions.
   - Use equality/reduce voting only if multiple samples are needed for robustness.
4. Build a 7x7 integer grid using fixed masks:
   - level0 border mask: rows 0/6 or cols 0/6
   - level1 border mask: rows 1/5 or cols 1/5 inside
   - level2 border mask: rows 2/4 or cols 2/4 inside
   - center: row 3 col 3
5. Convert to required one-hot output format only at the final step if necessary.
6. Pad 7x7 to [1,10,30,30] only after construction.

Hard constraints:
- Do not use Python runtime logic in inference; produce a pure ONNX graph.
- Do not overbuild a generic ARC solver.
- Do not use large lookup tables for all pixels.
- Do not store a full 7x7x10 constant if avoidable; generate masks arithmetically or with tiny coordinate constants.
- Benchmark memory and params with NeuroGolf scoring, then iterate until cost is minimal.
- Try multiple variants:
  A) fixed coordinate sampling + tiny template masks
  B) direct gather of 4 colors + nested Where construction
  C) precomputed 7x7 level-index grid with Gather(colors, level_index)
- Keep the variant with the lowest cost, not the prettiest graph.

Expected output pattern:
7x7 nested square:
outer color on border,
second color on ring 1,
third color on ring 2,
center color at (3,3).

--------------------------------------

tohle je task097 na vstupu jsou boxy stejne barvy na vystupu zustanou pouze ty ktere maji v dosahu jednoho bloku dalsi blok. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Přemýšlení několik sekund

Použij tenhle prompt pro Codex/Cursor:

We need solve NeuroGolf ARC task097 with an optimized ONNX model.

Task:
Input/output shape is always 16x11 -> 16x11.
Grid has black background 0 and one foreground color, usually cyan/teal.
Output keeps only foreground pixels that have at least one other foreground pixel within Chebyshev distance 1.
In other words, keep a colored cell if any of its 8 neighboring cells is also the same nonzero foreground color.
Remove isolated single pixels.
Preserve original positions and original foreground color.

Examples:
- adjacent horizontal/vertical/diagonal pairs or clusters remain
- any single pixel with no neighbor in the surrounding 3x3 window becomes 0
- background remains 0

Goal:
Create the smallest / lowest NeuroGolf-cost ONNX possible.
Optimize for score, not readability. File size alone does not matter unless it reduces params or internal tensor memory.

Implementation strategy:
Input is one-hot ARC tensor [1,10,30,30], but useful region is only 16x11.
Extract/crop the [16,11] active area.
Use channel-wise or color-agnostic nonzero mask:
  fg = sum channels 1..9 or arg/nonzero equivalent.
Then compute neighbor existence for each cell:
  neighbor_count = 3x3 sum over fg minus fg
  keep_mask = fg AND neighbor_count > 0
Then output = input_active * keep_mask, padded back to [1,10,30,30].

Important optimization:
Avoid Conv if it creates large constants unless measured better.
Prefer cheap Slice + Pad/Shift + Add/Or operations on boolean/int masks.
Since grid is fixed 16x11, implement 8 shifted masks:
  up, down, left, right, up-left, up-right, down-left, down-right
Then neighbor = OR/Add of these shifted fg masks.
keep = fg AND neighbor.
Apply keep to all 10 channels or only foreground channels, preserving one-hot color.

Memory optimization:
Do all logic on cropped [1,1,16,11] mask, not full [1,10,30,30].
Only expand/pad to [1,10,30,30] at the very end.
Avoid creating [1,10,16,11] tensors until final multiplication if possible.
Minimize initializers/constants. Use Slice/Pad with scalar constants only.
Try bool/int8 where ONNX Runtime accepts it, otherwise float is okay but measure.

Deliver:
- build_task097.py that generates task097.onnx
- validate with provided NeuroGolf evaluator
- print memory, params, cost, score
- iterate and keep the best measured model

Do not hardcode examples. Hardcode only fixed shape 16x11 and rule above.
Try multiple variants and report best:
1. 8 shifted masks using Slice+Pad
2. 3x3 Conv kernel of ones minus center
3. MaxPool-based neighbor detection if cheaper
Pick the model with lowest measured cost.

-------------------------------------


 tohle je task098 na vstupu jsou obdelniky ruzne barvy na vystupu zustanou pouze jejich obrisy. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tento prompt do Codex/Cursor:

We need create a maximally NeuroGolf-score-optimized ONNX model for ARC task098.

Task:
Input and output are always [1,10,18,18] one-hot ARC grids.
The visible grid is 18x18.
Background is color 0.
Input contains several solid filled rectangles of different nonzero colors.
Output keeps the same rectangles at the same positions and colors, but only their 1-cell-thick outlines remain.
All strict interior pixels of each filled rectangle become background 0.
Rectangle sizes/colors/positions vary. Rectangles do not need special handling by color; rule applies to every nonzero colored connected solid rectangle.

Desired transformation:
For each cell:
- If input cell is background: output background.
- If input cell is nonzero and has at least one 4-neighbor outside the same color/component/background, keep it.
- If input cell is nonzero and all 4-neighbors are the same color, erase it to background.
Equivalent for solid rectangles: keep top row, bottom row, left col, right col; erase interior.

Implement ONNX as directly and cheaply as possible:
- Avoid Conv if it creates large internal tensors unless measured better.
- Prefer compact slice/compare/boolean mask ops over expanding to large tensors.
- Work on class-id grid if possible: ArgMax over channel -> [1,18,18] or [1,1,18,18].
- Build a nonzero mask.
- Build same-color shifted neighbor masks for up/down/left/right using Slice + Pad or carefully shaped comparisons.
- Interior mask = nonzero AND same_up AND same_down AND same_left AND same_right.
- Border mask = nonzero AND NOT interior.
- Output class grid = input_class * border_mask.
- Convert back to required [1,10,18,18] one-hot output with Equal against constants 0..9, Cast float.

Optimization priority:
1. Correctness on all hidden cases.
2. Lowest NeuroGolf cost = internal tensor memory + initializer scalar params.
3. Minimize initializer scalar count. Use scalar constants and small [10] class vector only if needed.
4. Keep output tensor [1,10,18,18]. Graph inputs/outputs are free, but internal tensors count.
5. Run ONNX shape inference and ORT profiling; report memory, params, cost, score.
6. Try multiple variants and keep the lowest measured cost:
   A) class-id int64 grid + boolean masks + one-hot at end.
   B) per-channel direct mask if cheaper.
   C) crop-center method: compute interior only on [1,16,16], pad border false.
   D) use MaxPool/Conv only if measured lower, not assumed.

Important implementation hint:
Because rectangles are solid same-color objects, a pixel is interior iff:
grid[y,x] != 0
AND grid[y,x] == grid[y-1,x]
AND grid[y,x] == grid[y+1,x]
AND grid[y,x] == grid[y,x-1]
AND grid[y,x] == grid[y,x+1]
for valid inner positions. Outer grid boundary can never be interior.

Use only ONNX ops supported by onnxruntime. No custom ops.
Create a script build_task098.py that writes task098.onnx and a benchmark script that verifies the shown sample plus randomized solid-rectangle tests.
Do not optimize file size; optimize NeuroGolf measured cost.

--------------------------------------------------------------

tohle je task99 na vstupu je modry obdelnik ktery uprostred ma jeden blok jine barvy kterou je vyplnen s jednou radou nad obdelnikem. vstup je vzdy 10x10.Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt pro Codex/Cursor:

We need solve NeuroGolf ARC task099 with a hand-optimized ONNX model, maximizing score = minimize internal tensor memory + initializer params.

Task:
Input is always [1,10,10] or one-hot equivalent [1,10,30,30] depending existing harness; logical grid is always 10x10.
Output is also 10x10.

Transformation:
There are blue rectangular/outline objects. Each object contains exactly one non-blue colored seed cell inside or in its hole region.
For each blue rectangle/object:
1. Keep the blue border/shape unchanged.
2. Fill the interior/connected hole area of that blue object with the seed color.
3. Also add one full horizontal row of the seed color directly above the top edge of that blue rectangle/object, spanning the rectangle width.
Background stays 0.

Example:
Top blue object has red seed inside. Output fills the inside with red and adds a red row above the blue rectangle.
Bottom blue object has green seed. Output fills its inner area with green and adds green row above it.

Important constraints:
- Input size fixed 10x10, so prefer hardcoded slicing/indexing over generic dynamic algorithms.
- Optimize for NeuroGolf score, not file size.
- Avoid large internal tensors. Do not create full one-hot [10 colors,10,10] intermediates unless strictly necessary.
- Prefer scalar masks, small slices, Gather/Slice/Where, Add/Mul, ReduceMax over expensive expanded tensors.
- Initializers count as params, so use as few constants as possible.
- Output tensor memory is free, but internal tensors are counted.
- If final Kaggle harness requires [1,10,30,30], build compact 10x10 first and pad/expand only at the final graph output.

Implementation strategy:
- Work on compact 10x10 integer grid.
- Detect blue cells as input == 1.
- Detect non-background non-blue seed cells as input != 0 and input != 1.
- Since grid is fixed 10x10, derive each object bounding box cheaply:
  - rows/cols containing blue cells.
  - split components if needed by connected blue groups / separated row ranges.
- For each blue component:
  - top = min row of its blue cells
  - bottom = max row
  - left = min col
  - right = max col
  - seed color = nonzero nonblue cell inside [top:bottom+1, left:right+1]
  - fill area:
    rows top+1..bottom-1 and cols left+1..right-1 where original is background or seed -> seed color
  - add row top-1, cols left..right = seed color
  - preserve blue cells as blue.
- Multiple components can exist, usually 1-2. Handle both independently.

Need deliverable:
- Python script that creates task099.onnx.
- Include a local verifier with the shown example and at least 2 synthetic cases.
- Print NeuroGolf-relevant stats: initializer scalar count, inferred internal tensor shapes, and estimated memory.
- Try several implementations and keep the lowest cost:
  A) direct hardcoded coordinate extraction for known 10x10 pattern structure
  B) generic bbox from blue masks
  C) morphology-like fill using row/col ranges
- Save best model as task099.onnx.

Optimization target:
Use the smallest possible internal tensors. A correct model with slightly hardcoded 10x10 logic is preferred over a generic elegant model with large intermediates.

--------------------------------------------

tohle je task100 vstup je vzdy 10x10. na vstupu jsou dva obdelniky jeden je vetsi jeho barva je dana na vystup 2x2. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Create the most score-optimized ONNX model for NeuroGolf ARC task100.

Task:
- Input is always 10x10 one-hot ARC grid `[1,10,10,10]`.
- Output must be `[1,10,2,2]`.
- Input contains exactly two hollow rectangles of different colors.
- One rectangle is larger than the other.
- Output is a solid 2x2 block filled with the color of the larger rectangle.
- Background is color 0.
- Ignore the smaller rectangle completely.

Optimization goal:
- Maximize NeuroGolf score, not readability.
- Minimize `cost = params + internal tensor memory`.
- Constants count as params, so avoid large constants and avoid lookup tables.
- Internal tensors are expensive; keep all intermediates as small as possible.
- Graph input/output tensors are free.
- Prefer scalar/very small reductions over full 10x10 materialization.
- Do not optimize for ONNX file size unless it also reduces params or memory.

Suggested ONNX logic:
1. Extract non-background channels `1..9`.
2. For each color channel, compute object area/pixel count over 10x10:
   - `cnt[c] = ReduceSum(input[:, c:c+1, :, :])`
3. Since each rectangle is hollow, the larger rectangle has larger colored-pixel count.
4. Find color with maximum count:
   - `argmax_c = ArgMax(cnt over colors 1..9)`
5. Convert selected color to one-hot output channel:
   - produce `[1,10,1,1]` one-hot vector with selected color channel = 1, all others 0.
   - channel 0 must stay 0.
6. Expand/broadcast only at the very end to `[1,10,2,2]`.

Important:
- Avoid creating `[1,10,10,10]` intermediate masks except views/slices from input.
- Best graph likely uses:
  `Slice/Gather -> ReduceSum -> ArgMax -> Equal/OneHot -> Cast -> Expand`
- Try both variants and measure:
  A) `OneHot(ArgMax+1)` then reshape/expand to 2x2.
  B) Compare `[0..9]` channel ids against selected color using a tiny constant `[10]`.
- Use the variant with lower NeuroGolf cost after actual scoring.
- Use smallest possible dtypes for constants if accepted by ORT.
- Final output must be float/one-hot compatible with the evaluator.

Expected behavior example:
- If input has orange hollow rectangle and larger cyan hollow rectangle, output is 2x2 solid cyan.

-----------------------------------------------------

tohle je task101 na vstupu je patern ktery je doplnen na mista kde je patern nedokonceny s tim ze muze byt zvetseny. Patern je modry s cervenou casti cervena cast urcuje pozici ostatnich.  Dej mi prompt pro vytvoreni onnx s nejlespim moznym score 
Přemýšlení několik sekund

Použij tenhle prompt pro Codex/Cursor:

Create a highly optimized ONNX solver for NeuroGolf ARC task101.

Task:
- Input/output size is always 17x21.
- Background is 0.
- There is one complete blue pattern with a red marker cell/area inside it.
- Elsewhere in the grid there are red marker cells/areas that indicate where the same pattern must be generated.
- The target pattern can appear at the original scale or enlarged.
- Red pixels define the anchor/relative position of the blue pattern. For every red marker object, reconstruct the corresponding full blue pattern around it.
- Keep all original red pixels unchanged.
- Fill only the missing blue pixels needed to complete each pattern instance.
- If the marker is larger, scale the blue pattern by the same factor. Example: a 2x2 red block corresponds to a 2x enlarged copy of the original pattern.
- Output is input plus all completed blue pattern copies.

Optimization goal:
- Maximize NeuroGolf score, not readability.
- Minimize internal tensor memory and initializer params.
- Do not optimize for ONNX file size unless it reduces params or internal tensors.
- Graph input/output tensors are free, but internal tensors are expensive.
- Avoid large one-hot tensors, avoid full [1,10,17,21] intermediates if possible.
- Prefer int64/bool masks only where necessary; use small constants.
- Use Slice/Gather/Equal/Where/Or/And/Mul/Add/Reduce ops efficiently.
- Any opset is allowed if it improves measured score.

Implementation strategy:
1. Load the 17x21 input grid.
2. Detect all non-background pixels:
   - blue = 1
   - red = 2
3. Identify the complete source pattern:
   - Find the connected blue component that contains/has adjacent red marker inside its bounding box.
   - Compute its bounding box.
   - Compute red marker position relative to that bounding box.
4. For each red connected component:
   - Compute its bbox and scale factor from its size compared to the source red marker.
   - Use red bbox as anchor.
   - Recreate the source blue mask at that scale around the red marker anchor.
5. Merge all generated blue masks with the original grid:
   - Put blue where generated mask is true and current cell is background.
   - Preserve red cells.
6. Return [1,10,30,30] only if required by evaluator; otherwise return compact 17x21 if accepted. If padding is required, pad only at the final output.

Important:
- Hardcode input size 17x21.
- Hardcode color ids if useful: 0 background, 1 blue, 2 red.
- Use exhaustive validation against provided train examples.
- After building ONNX, run shape inference and ORT profiling.
- Report memory, params, cost, and NeuroGolf score.
- Try multiple graph variants and keep the one with lowest cost.

Generate:
- A self-contained Python script that builds the ONNX model.
- Hardcoded paths, no argparse.
- Include validation against local task101 JSON if available.
- Include scoring/profiling helper.

Silnější verze pro optimalizaci:

Now aggressively golf the ONNX graph for task101. Try at least 5 variants:
A) direct coordinate-mask construction
B) bbox/anchor reconstruction
C) convolution-neighborhood detection of source pattern
D) precomputed coordinate constants with Where
E) minimal branching using broadcasted row/col grids

For each variant measure:
- params
- internal memory
- cost
- score

Keep only the best-scoring ONNX. Do not assume opset 10. Use whatever opset gives the smallest measured NeuroGolf cost.

---------------------------------------------------

tohle je task103 na vstupu je cerveny 3x3 patern vystup je bud modra nebo oranzova barva. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Create the most score-optimized ONNX solver for ARC/NeuroGolf task103.

Task rule:
- Input is always a 3x3 grid.
- Relevant colors:
  - 0 = black/background
  - 2 = red
- Output is always a 1x1 grid.
- The output color is either:
  - 1 = blue
  - 7 = orange
- Decide the output from the red/black 3x3 pattern.

Observed example:
Input:
2 0 2
0 2 0
2 0 2
Output:
1

This example is the “X / both diagonals” pattern, so return blue.

Implementation goal:
- Build a tiny deterministic ONNX graph.
- No ML, no Conv, no large internal tensors.
- Input/output format must match NeuroGolf expected tensor format, but keep internal tensors as small as possible.
- Input tensor may be `[1,10,30,30]`; only read top-left 3x3 from channel 0.
- Final output should be padded/placed into required output format only at the very end.
- Since graph inputs/outputs are free but internal tensors are expensive, avoid creating `[1,10,30,30]` intermediates.

Preferred ONNX logic:
1. Slice the 3x3 active grid from input channel 0.
2. Convert red cells to boolean/int mask: `is_red = (x == 2)`.
3. Compute compact pattern features only from the 9 cells:
   - center red
   - four corners red count
   - four edge-centers red count
   - main diagonal count
   - anti-diagonal count
   - total red count
4. Classify:
   - if center is red AND all four corners are red AND edge-centers are black, output blue `1`.
   - otherwise output orange `7`.
5. Emit 1x1 logical result, then only pad/expand final tensor to NeuroGolf-required output shape if necessary.

Optimization requirements:
- Minimize `cost = memory + params`.
- Avoid initializers where possible; use scalar Constants only when unavoidable.
- Prefer Slice/Gather/Equal/And/ReduceSum/Where on tiny tensors.
- Avoid OneHot, Tile, Conv, MatMul, large Cast chains, or any operation producing large intermediates.
- Do not optimize for ONNX file size; optimize NeuroGolf score.
- After creating the model, run shape inference and ORT profiling to report:
  - internal memory
  - params
  - total cost
  - score
- Try alternative implementations and keep the one with lowest measured cost.

Deliver:
- `task103.onnx`
- Python generator script
- validation script testing the shown example and several invented 3x3 cases.

-----------------------------------------------

tohle je task104 vstup je vzdy 3x3 vystup 9x9 dve kostky smerem cerveneho stredu. task by mel mit jen 4 mozne inputy. Dej mi prompt pro vytvoreni nejlepsiho onnx.

Použij tenhle prompt pro Codex/Cursor:

We need create the smallest/highest-score ONNX solver for NeuroGolf ARC task104.

Task:
- Input is always a 3x3 ARC grid, one-hot or integer grid depending on the existing runner.
- Output must be 9x9.
- Colors:
  - background = 0
  - green = 3
  - red = 2
- The red cell is always the center input cell at (1,1).
- There are exactly two green cells adjacent/diagonal around the red center.
- Only 4 possible input variants exist.
- Output contains two 4x4 green blocks placed in the 9x9 grid in the direction of the two green cells relative to the red center.
- Red is not copied to output.

Rule:
For each green cell around center:
- if green is top-left of center -> draw green 4x4 block at output rows 0..3, cols 0..3
- if green is top-right -> draw green 4x4 block at rows 0..3, cols 5..8
- if green is bottom-left -> draw green 4x4 block at rows 5..8, cols 0..3
- if green is bottom-right -> draw green 4x4 block at rows 5..8, cols 5..8

Examples:
Input:
[[3,3,0],
 [3,2,0],
 [0,0,0]]
Output:
green 4x4 block at top-left and another 4x4 block at bottom-right? Wait, derive from occupied corner direction around red:
the 2x2 green+red quadrant indicates the diagonal direction. Implement exactly by detecting which of the four 2x2 quadrants around the center contain 3 green cells + red center, then emit the corresponding 4x4 quadrant block.

Better robust rule:
Compute four quadrant indicators:
TL = input[0,0]==3 OR input[0,1]==3 OR input[1,0]==3
TR = input[0,2]==3 OR input[0,1]==3 OR input[1,2]==3
BL = input[2,0]==3 OR input[2,1]==3 OR input[1,0]==3
BR = input[2,2]==3 OR input[2,1]==3 OR input[1,2]==3

But because each sample is a 2x2 L shape around the red center, only two opposite/related output blocks are active:
- top-left L around center -> output blocks TL and BR
- top-right L around center -> output blocks TR and BL
- bottom-left L around center -> output blocks BL and TR
- bottom-right L around center -> output blocks BR and TL

Implement the direct minimal mapping from the four possible L-shapes:
case A: x00=3, x01=3, x10=3 -> output TL 4x4 and BR 4x4
case B: x01=3, x02=3, x12=3 -> output TR 4x4 and BL 4x4
case C: x10=3, x20=3, x21=3 -> output BL 4x4 and TR 4x4
case D: x12=3, x21=3, x22=3 -> output BR 4x4 and TL 4x4

Optimization goal:
- Minimize NeuroGolf cost = params + internal tensor memory.
- Graph input/output tensors are free, internal tensors are expensive.
- Avoid creating full [1,10,9,9] intermediates if possible.
- Prefer tiny scalar/slice logic + final constant masks.
- Constants count as params, so keep constants tiny.
- Avoid large Constant 9x9 templates unless it still scores better after measurement.
- Try multiple ONNX variants and measure each with the official scorer. Keep the lowest cost.

Suggested implementation strategy:
1. Extract only the relevant scalar cells from the 3x3 input:
   x00,x01,x02,x10,x12,x20,x21,x22, probably channel/color-3 if one-hot.
2. Build boolean case indicators using Equal to green and And:
   A = green(x00)&green(x01)&green(x10)
   B = green(x01)&green(x02)&green(x12)
   C = green(x10)&green(x20)&green(x21)
   D = green(x12)&green(x21)&green(x22)
3. Because A and D both map to TL+BR, and B and C map to TR+BL:
   diag1 = A OR D
   diag2 = B OR C
4. Produce a 9x9 integer output:
   rows 0..3, cols 0..3 = 3 if diag1
   rows 5..8, cols 5..8 = 3 if diag1
   rows 0..3, cols 5..8 = 3 if diag2
   rows 5..8, cols 0..3 = 3 if diag2
   everything else = 0.
5. If output required shape is [1,10,30,30], pad/cast only as final step, not as an internal full-size tensor if avoidable.

Important:
- Do not use Conv.
- Do not use large intermediates.
- Do not store a huge lookup table unless measured better.
- Prefer static Slice/Gather/Equal/And/Or/Mul/Add/Concat/Pad.
- Benchmark variants:
  A) boolean diag1/diag2 + construct rows via Concat
  B) tiny Gather + 2 constant 9x9 masks multiplied by diag flags
  C) full 4-case OneHot lookup only if params are still acceptable
- Print for every candidate: params, memory, cost, score, and ONNX node count.
- Return the best measured model.

Expected final behavior:
Input:
[[3,3,0],
 [3,2,0],
 [0,0,0]]
Output has green 4x4 blocks at top-left and bottom-right.

Input:
[[0,3,3],
 [0,2,3],
 [0,0,0]]
Output has green 4x4 blocks at top-right and bottom-left.

Největší šance na dobré score: nejdřív zkus variantu se 2 maskami 9x9, protože bude jednoduchá a málo runtime tensorů; pokud params moc bolí, přepiš na konstrukci řádků přes malé Concat.


---------------------------------------------

tohle je task105 na vstupu je modru obdelnik s mezerama ktery muse mit caru uprostred vertikalne i horizontalne ktery je na vystup doplneny cervenou.Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create a NeuroGolf/ARC-AGI ONNX solver for task105 with maximum score, not maximum readability.

Task:
- Input/output grid size is always 13x13.
- Background is 0.
- Input contains a blue (1) rectangular outline-like shape with gaps.
- Output keeps all original blue cells and fills the missing cells of the rectangle’s central horizontal and central vertical lines with red (2).
- In other words: find the bounding box of all blue pixels, compute its middle row and middle column, and paint red only on currently empty cells inside the rectangle on those two center lines. Existing blue stays blue. Everything else stays 0.
- Expected output is same 13x13 grid.

ONNX optimization goal:
- Minimize NeuroGolf cost = internal tensor memory + params.
- Inputs/outputs may be Kaggle one-hot `[1,10,30,30]`, but internal computation must crop/use only 13x13 or smaller.
- Final padding to `[1,10,30,30]` only at the very end if required by evaluator.
- Prefer int/bool ops and constants with smallest possible tensors.
- Avoid Conv and large one-hot intermediates.
- Avoid full `[1,10,30,30]` tensors except final output.
- Do not hardcode one training example; solve all possible 13x13 positions/sizes consistent with the rule.

Implementation strategy:
1. Extract class-1 plane from input top-left 13x13.
2. Build boolean mask `blue = (x == 1)`.
3. Compute row occupancy and column occupancy using ReduceMax/ReduceSum over 13 cells.
4. Find bounding box:
   - top = first occupied row
   - bottom = last occupied row
   - left = first occupied col
   - right = last occupied col
5. Compute:
   - mid_row = (top + bottom) // 2
   - mid_col = (left + right) // 2
6. Build two boolean masks over 13x13:
   - horizontal: row == mid_row and left <= col <= right
   - vertical: col == mid_col and top <= row <= bottom
7. `cross = horizontal OR vertical`.
8. Output grid:
   - blue cells remain value 1.
   - empty cells where `cross` is true become value 2.
   - otherwise 0.
   Formula: `out = Where(blue, 1, Where(cross, 2, 0))`.
9. Convert back to required output format.

Score work:
- Generate several ONNX variants and benchmark NeuroGolf score.
- Try opsets freely; do not force opset 10.
- Compare:
  A) generic bbox/midline implementation,
  B) precomputed 13x13 row/col index constants,
  C) possible lookup/MatMul-free version,
  D) any specialized variant exploiting fixed 13x13.
- Keep the valid model with lowest `memory + params`.
- Print memory, params, cost, score, and save best model as `task105.onnx`.

Use a self-contained Python script with hardcoded paths, no argparse.

--------------------------------------

tohle je task106 na vstupu je2x2 patern z ktereho je udelna 4x4 vystup tim ze se pootaci o 90 stupnu. Dej mi prompt pro vytvoreni optimalizovaneho onnx s nejlepsim moznym score
Create the most NeuroGolf-score-optimized ONNX model for ARC task106.

Task:
- Input grid is always 2x2.
- Output grid is always 4x4.
- The 2x2 input pattern is expanded into a 2x2 tiling of four 2x2 blocks.
- Each output quadrant is the original input rotated by a multiple of 90 degrees:
  top-left     = input rotated 0°
  top-right    = input rotated 90° clockwise
  bottom-left  = input rotated 90° counterclockwise
  bottom-right = input rotated 180°

For input:
[[a, b],
 [c, d]]

Expected 4x4 output:
[[a, c, c, d],
 [b, d, a, b],
 [b, a, d, b],
 [d, c, c, a]]

Important:
- Optimize purely for NeuroGolf score, not file size.
- Minimize internal tensor memory and initializer scalar count.
- Graph input/output tensors are free, but internal tensors are expensive.
- Avoid building large intermediate tensors.
- Prefer direct Slice/Gather/Concat construction of the 4 output rows.
- Use constants only when they reduce total cost.
- Try multiple ONNX variants and measure actual NeuroGolf score.
- Do not force a specific opset unless it gives a better measured score.
- Final ONNX must pass shape inference and ONNX Runtime validation.

Implementation strategy:
1. Extract the four scalar cells a,b,c,d from the 2x2 input.
2. Construct each 4-element output row directly:
   r0 = [a,c,c,d]
   r1 = [b,d,a,b]
   r2 = [b,a,d,b]
   r3 = [d,c,c,a]
3. Concatenate rows into a 4x4 output.
4. If required by NeuroGolf wrapper, pad/reshape only at the final output stage, never earlier.
5. Benchmark alternatives:
   - scalar Slice + Concat
   - Gather with constant indices + Reshape
   - one flattened Gather from input.reshape(4) using indices [0,2,2,3,1,3,0,1,1,0,3,1,3,2,2,0]
6. Choose the graph with the lowest `memory + params`.

Goal:
Produce the valid ONNX with the best possible NeuroGolf score for task106.

-------------------------------------------


tohle je task107 na vstupu je ctverec z jehoz rohu na vystupu vychazi diagonalne cervena. Dej mi prompt pro vytvoreni optimalizovaneho onnx s nejlepsim moznym score

Použij tenhle prompt do Codex/Cursor:

We need create the most score-optimal ONNX model for NeuroGolf ARC task107.

Task:
Input is always 5x5, output is always 20x20.

Observed rule:
- The 5x5 input contains colored blocks forming a small 5x5 pattern.
- Output is a 4x scaled/tiled representation:
  each input cell becomes a 4x4 block in the output at the corresponding position.
- Additionally, from the corners of the central blue square, red diagonal rays are drawn outward on the 20x20 output grid.
- Red diagonals have color value 2.
- Existing scaled colored regions remain, except red diagonal cells overwrite black/background cells where needed.
- The main transformation is:
  out[y*4:(y+1)*4, x*4:(x+1)*4] = input[y,x] for every input cell.
  Then draw red diagonals from the four corners of the scaled central square:
    top-left corner ray goes up-left,
    top-right corner ray goes up-right,
    bottom-left corner ray goes down-left,
    bottom-right corner ray goes down-right.
- For the shown example, the central 2x2 blue input block at rows 1..2, cols 1..2 becomes an 8x8 blue square at output rows 4..11, cols 4..11, and red diagonals start from its four outer corners.

Optimization goal:
- Maximize NeuroGolf score, not readability.
- Minimize params + internal tensor memory.
- Inputs/outputs are free, internal tensors are expensive.
- Avoid large full 20x20 intermediate tensors unless absolutely necessary.
- Prefer Slice/Gather/Reshape/Concat/Expand/Tile only if measured cheaper.
- Hardcode constants aggressively if it reduces params/memory.
- Do not force a specific opset; choose whichever gives lowest measured NeuroGolf cost.

Implementation strategy:
1. Build ONNX directly with onnx.helper.
2. Input shape should match required Kaggle/NeuroGolf format, usually [1,10,30,30].
3. Slice/extract only the active 5x5 area from channel 0.
4. Produce active 20x20 output.
5. Cheapest likely approach:
   - Use Resize/Expand/Tile/Reshape to repeat each 5x5 cell into 4x4 block if it scores well.
   - Alternative: manually Slice each needed row/cell and Concat repeated values if that has lower params/memory.
6. Add fixed red diagonal mask for the four rays:
   - Construct a constant 20x20 uint/int mask with 2 on diagonal ray cells and 0 elsewhere.
   - But compare cost: constant mask has 400 params, maybe too expensive.
   - Better: generate diagonal positions using cheap constants and ScatterElements/Where only if lower total cost.
   - Since only a few red cells are needed, prefer small index constants + ScatterND over full 20x20 constant.
7. Combine:
   - background/scaled = nearest-neighbor 4x expansion of input 5x5.
   - diag_mask sets selected diagonal cells to red.
   - output_active = Where(diag_mask_nonzero, 2, scaled).
8. Pad active 20x20 to required [1,10,30,30] output only at the final step.

Important:
- Measure NeuroGolf cost after every variant.
- Try at least these variants:
  A) Resize nearest 5x5 -> 20x20 + ScatterND red diagonals.
  B) Reshape/Expand/Reshape repeat cells 4x4 + ScatterND red diagonals.
  C) Fully manual Concat construction with no large internal 20x20 until final.
  D) If all inputs share same colored layout except colors, exploit that and hardcode geometry.
- Report memory, params, cost, and score for each variant.
- Keep the best-scoring ONNX only.
- Validate exact output on all train examples.

----------------------------------------------

tohle je task108 vstup je vzdy 10x10 vystup 20x20 kazdy z kazdeho bloku na vstupu je udelano 4x4 na vystupu. dej mi prmopt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt pro Codex/Cursor:

Create the most score-optimized ONNX model for NeuroGolf ARC task108.

Task rule:
- Input is always one-hot ARC grid tensor with shape [1,10,10,10].
- Logical grid size is exactly 10x10.
- Output must be Kaggle/NeuroGolf-compatible tensor [1,10,30,30], but only the top-left 20x20 region is meaningful.
- Transformation: every nonzero input cell becomes a solid 4x4 block of the same color in the output.
- Background remains 0.
- So logical output is 20x20, created by expanding each input pixel by factor 2 in height and width? IMPORTANT: based on examples, each original grid cell becomes a 4-cell area in 20x20, i.e. a 2x2 spatial expansion, not 4x4 in coordinate size. If the visual description says “4x4”, verify against examples: 10x10 -> 20x20 means scale factor 2.
- Final output must be padded with zeros from [1,10,20,20] to [1,10,30,30].

Optimization target:
- Maximize NeuroGolf score = minimize cost = internal tensor memory + params.
- Inputs/outputs are free; internal tensors are expensive.
- Avoid large internal tensors where possible.
- Avoid trainable initializers; prefer Constant nodes only if necessary and tiny.
- Keep params near zero.
- Use simple ONNX ops only.

Best likely implementation:
1. Take input [1,10,10,10].
2. Spatially expand H and W by factor 2:
   - Prefer Resize with scales [1,1,2,2] and nearest mode if accepted and cheap.
   - Or use Unsqueeze/Expand/Reshape pattern if Resize creates lower params/memory.
3. Pad expanded [1,10,20,20] to [1,10,30,30] with zeros.
4. Output float32 one-hot tensor.

Important:
- Do not create intermediate [1,10,30,30] until the final output.
- Do not use Conv/MatMul.
- Do not use big constants.
- Measure with the NeuroGolf scoring/profiling script, not ONNX file size.
- Try multiple variants and keep the one with lowest measured cost:
  A) Resize nearest + Pad.
  B) Unsqueeze/Expand/Reshape + Pad.
  C) Tile-style expansion if cheaper.
- Print memory, params, cost, and score for each variant.
- Save best model as task108_best.onnx.

-------------------------------------------------

tohle je task109 vstup je rozdeleny na 4 casty v levem hornim rohu je patern ktery je na vystup pootaceny do vsech 4 casty v barve rezdeleni vstupu. Dej mi prompt pro vytvoreni onnx s nejlepsi moznym score
You are optimizing NeuroGolf ARC-AGI task109. Create the smallest/highest-score ONNX model possible.

Task:
- Input grid is always 11x11.
- Output grid is always 10x10.
- Input is divided by a red cross:
  - full red horizontal line at row 5
  - full red vertical line at col 5
- In the top-left 5x5 quadrant there is a blue pattern.
- Output removes the separator cross and creates a 10x10 grid made of four 5x5 quadrants.
- The blue pattern from input top-left must be copied into all four output quadrants, rotated around the center:
  - output top-left = original 5x5 pattern
  - output top-right = original rotated 90° clockwise
  - output bottom-right = original rotated 180°
  - output bottom-left = original rotated 270° clockwise
- All output pattern pixels must be recolored from blue to red.
- Background stays 0.

Implementation target:
- Produce ONNX directly, not a PyTorch model.
- Optimize for NeuroGolf score, not readability.
- Minimize internal tensor memory and params.
- Avoid large intermediate tensors.
- Graph input/output may be large/free, but internal tensors are expensive.
- Prefer Slice/Gather/Transpose/Concat/Where/Cast style ops.
- Do not store the 10x10 output as a Constant.
- Do not use unnecessary one-hot expansion.

Suggested graph:
1. Slice input channel/grid to get top-left 5x5 pattern.
2. Convert nonzero blue mask to red values:
   `P = Where(pattern != 0, 2, 0)` if ARC red is color 2.
3. Create rotations cheaply:
   - P0 = P
   - P90 = transpose + reverse rows/cols via Gather indices [4,3,2,1,0]
   - P180 = reverse both axes
   - P270 = transpose + opposite reverse
4. Build:
   - top = Concat(P0, P90, axis=width)
   - bottom = Concat(P270, P180, axis=width)
   - out10 = Concat(top, bottom, axis=height)
5. Pad/reshape only if Kaggle/NeuroGolf wrapper requires fixed `[1,10,30,30]`; otherwise output exact 10x10.

Try multiple variants and report measured:
- params
- internal memory
- cost
- score
- ONNX ops used
- whether shape inference/profiling creates extra intermediates

Goal:
Find the absolute lowest-cost valid ONNX. Hard-optimize every op and initializer.

----------------------------------------

tohle je task110 na vstupu je opakujici se patern s mezerama ktere jsou na vystupu vyplnene. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create an ultra-minimal ONNX solver for NeuroGolf ARC task110.

Task:
- Input and output are always 29x29.
- Input is an almost fully filled repeating periodic color pattern.
- Some cells are black/0 holes.
- Output is the same 29x29 grid with all black holes filled by continuing the repeating pattern.
- Existing nonzero cells must stay unchanged.

Goal:
- Maximize NeuroGolf score, not readability.
- Minimize internal tensor memory and initializer params.
- Do not pad or create large unnecessary intermediates.
- Use any ONNX opset only if it improves measured score.

Important pattern observation:
- The whole grid is generated by a small periodic tile.
- Solve by reconstructing each zero cell from same-phase nonzero cells modulo the tile period.
- Likely periods are small; test periods such as 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.
- Choose the smallest period pair `(py, px)` that is consistent with all visible nonzero cells.
- For task110, prefer hardcoding the discovered best period after testing on examples.

Implementation strategy:
1. Input shape is `[1, 10, 29, 29]` one-hot ARC format.
2. Extract class grid with ArgMax over channel axis → `[1,29,29]`.
3. Build zero mask: `grid == 0`.
4. Build a generated 29x29 template:
   - For every output coordinate `(y,x)`, use value from the canonical source coordinate `(y % py, x % px)`.
   - The canonical tile should be derived from nonzero cells in the input, preferably first available nonzero cell with same `(y % py, x % px)`.
   - If dynamic derivation is too expensive, hardcode the tile as tiny Constant initializers after inferring it from examples.
5. Output grid:
   - `out = Where(grid != 0, grid, template)`.
6. Convert back to one-hot `[1,10,29,29]`.

Optimization requirements:
- Avoid full one-hot internal tensors until the final output if possible.
- Keep internal tensors as `uint8`, `int64`, or `bool` only where needed.
- Prefer scalar/vector constants over large `[29,29]` constants unless a full constant template has lower measured cost.
- Benchmark variants:
  A. Hardcoded full `[29,29]` class template + Where + final one-hot.
  B. Hardcoded small tile + Gather/mod index expansion.
  C. Dynamic modulo voting/fill from input.
- Select by measured NeuroGolf cost, not by elegance.
- Remember: graph inputs/outputs are free, internal tensors and initializer scalar count are expensive.
- If full `[29,29]` template as initializer counts too many params, use small repeated tile instead.
- Final output must exactly match `[1,10,29,29]`.

Deliverables:
- Python script that writes `task110.onnx`.
- Include local validation against all train pairs.
- Print NeuroGolf-relevant stats: initializer scalar count, inferred internal tensor memory if available, and final score estimate.
- Try multiple graph variants automatically and keep the lowest measured cost model.


---------------------------------------

tohle je task 111 vstup je vzdy 10x10 . Na vstupu jsou ruzne paterny, jeden z nich u sebe ma sedy blok tenhle patern je dan na vystup. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt do Codex/Cursor:

We need solve NeuroGolf ARC task111 by generating a maximally score-optimized ONNX model.

Task rule:
- Input is always 10x10 ARC grid, likely shape [1,10,10,10] one-hot or equivalent existing NeuroGolf format.
- There are several blue patterns/objects on black background.
- Exactly one pattern is marked by an adjacent gray cell.
- Output is 3x3.
- Output must be the 3x3 blue/black pattern corresponding to the object that touches or is immediately adjacent to the gray marker.
- Gray is only a selector marker, not copied to output.
- Output colors: selected pattern keeps blue cells; all non-selected/background cells are black.
- In the shown example, the gray marker touches the upper pattern, so output is the 3x3 crop of that marked pattern.

Optimization goal:
- Do NOT build a generic connected-component solver unless it scores better.
- Hard-optimize for fixed 10x10 input and 3x3 output.
- Minimize NeuroGolf cost = internal tensor memory + params.
- Inputs/outputs are free; internal tensors are expensive.
- Prefer tiny boolean masks, Slice, Gather, Where, Add/Mul, ReduceMax/ReduceSum only if needed.
- Avoid large [1,10,10,10] internal tensors where possible.
- Avoid Conv unless it clearly reduces total cost.
- Use constants only when they reduce memory more than they add params.

Implementation strategy:
1. Extract color channels:
   - gray mask: input color 5
   - blue mask: input color 1
   - background is 0.
2. Find the gray cell position.
3. Determine which 3x3 candidate window/object is adjacent to the gray cell.
   Candidate windows are the 3x3 neighborhoods around possible object anchors near the gray marker.
4. Output the 3x3 blue pattern selected by the gray marker:
   - crop the selected 3x3 region from blue mask
   - convert True cells to blue color channel/output value
   - black elsewhere.
5. If using one-hot output [1,10,3,3], set channel 1 from selected blue mask and channel 0 as inverse; other channels zero.
   If scalar output is expected by local harness, output [1,3,3] int64/float as ARC color ids.
6. Validate against train examples and inspect inferred shapes/profiler memory.
7. Try multiple ONNX variants and report exact NeuroGolf memory/params/cost/score:
   - Variant A: gray-position based dynamic Slice/Gather crop
   - Variant B: enumerate all possible 3x3 crops and select using marker adjacency scores
   - Variant C: sparse constant index Gather if cheaper
8. Keep the final ONNX with the lowest measured cost, not the prettiest graph.

Important:
- The output is NOT all objects, only the object/pattern selected by the gray marker.
- The gray marker can touch the selected pattern from different sides, so selection must be based on adjacency/proximity to gray, not fixed coordinates.
- Fixed input size 10x10 and output size 3x3 should be exploited aggressively.
- Do not force a specific opset; use whichever opset produces the smallest measured NeuroGolf cost.

Největší riziko je zbytečně obecný component-labeling. Tady chceš spíš “najdi šedý pixel → vyber 3x3 crop modrého patternu kolem něj”.


-------------------------------------------------

tohle je task112 na vstupu je cerveny pattern s zelenou kostkou kolem ktere je patern vykopirovan s 90 stuponovim otocenim. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create a maximally NeuroGolf-optimized ONNX model for ARC task112.

Task:
- Input shape is always 20x30.
- Output shape is always 20x30.
- Background is 0.
- There is one red pattern, color 2.
- There is a green 2-cell vertical marker, color 3, directly below the source pattern.
- The output keeps the original red pattern and adds a rotated copy of that pattern below the green marker.
- The added copy is the original red pattern rotated by 90 degrees.
- The green marker stays unchanged.
- Everything else remains background.

Observed rule:
1. Locate the red object: all cells with value 2.
2. Extract its tight bounding box.
3. Locate the green marker: cells with value 3.
4. Treat the green marker as the pivot/separator between original and copied pattern.
5. Copy the red pattern below the marker, rotated 90 degrees clockwise.
6. Place the rotated pattern so that it is centered horizontally around the green marker column and starts below the green marker with the same gap relation as in the examples.
7. Preserve original red cells and green marker.

Optimization goal:
- Optimize strictly for NeuroGolf score, not file size.
- Minimize internal tensor memory and initializer params.
- Avoid large 20x30 float tensors internally when possible.
- Use integer/bool tensors where allowed.
- Prefer Slice/Gather/Where/Concat/Transpose/Reverse/Pad style operations.
- Avoid Conv, MatMul, large constants, one-hot grids, or dense coordinate maps unless they reduce measured cost.
- Final output may be 20x30, but intermediate tensors should stay close to the red pattern bbox size.

Implementation strategy:
- Build ONNX programmatically in Python.
- Input tensor: [1, 1, 20, 30] or equivalent ARC format expected by the NeuroGolf harness.
- Output tensor: same shape.
- Use Equal(input, 2) to get red mask and Equal(input, 3) for green marker.
- Compute red bbox from nonzero rows/cols.
- Slice red bbox.
- Rotate red patch 90 degrees clockwise using Transpose + Reverse/Slice with negative step if supported; otherwise use Gather with tiny constant indices only for the patch dimension.
- Compute green marker position from NonZero/ReduceMin/ReduceMax.
- Place rotated patch below green marker.
- Combine as:
  output = input with rotated red patch written where rotated_patch == 2.
- Since ONNX has no scatter-update cheaply in all opsets, test these alternatives and keep the lowest NeuroGolf cost:
  A) generate only local rotated patch and pad it to 20x30 with dynamic pads, then Where(rotated_mask, 2, input)
  B) use ScatterND for the rotated red cell coordinates only
  C) use Slice/Concat reconstruction by rows if cheaper for 20x30
- Measure score with the official NeuroGolf evaluator after each variant.

Important:
- Do not hardcode the exact visible example coordinates.
- It is acceptable to specialize to fixed input size 20x30 and colors 0/2/3.
- Prefer dynamic bbox/marker logic, but if training/test inputs always share the same pattern-size constraints, exploit that to reduce memory.
- Try multiple ONNX opsets and keep the model with lowest measured cost.
- Report:
  - memory
  - params
  - cost
  - score
  - exact ONNX ops used
  - why the chosen placement logic is valid

Deliverables:
- `task112.onnx`
- `build_task112.py`
- A small verifier that runs the model on the examples and prints exact-match accuracy.

-----------------------------------------------------


tohle je task113 na vstupu jsou vylneny prvni radky ktere jsou na vystup zrcadleny na spodni radky. Dej mi prompt pro vytvoreni onnx s durazen ma maximalni dosazitelne score

Použij tenhle prompt do Codex/Cursor:

We need solve NeuroGolf ARC task113 with a maximally score-optimized ONNX model.

Task rule:
- Input and output are always 10x5.
- The top rows contain a colored horizontal stripe pattern.
- Output keeps the input unchanged at the top.
- The same nonzero top stripe rows are mirrored to the bottom of the grid in reverse vertical order.
- Background remains 0.
- Example: if row0=red and row1=cyan, output row8=cyan and row9=red.

Goal:
Create the smallest/cheapest valid ONNX graph for NeuroGolf scoring, optimizing cost = internal tensor memory + params.
Prioritize score over readability and file size.

Implementation strategy:
- Input is one-hot `[1,10,10,5]` or ARC grid format as required by existing NeuroGolf runner.
- Extract only the original 10x5 grid.
- Produce output 10x5:
  - rows 0..4/actual top region: same as input where nonzero
  - bottom rows are vertical mirror of top nonzero rows.
- Avoid large intermediate tensors.
- Do not use Conv, large constants, or full one-hot expansion unless absolutely necessary.
- Prefer Slice/Gather/Concat/Reverse-style graph operations.
- Best simple graph:
  1. Slice input grid rows 0:2 if task always has exactly first two rows filled.
  2. Reverse those two rows vertically.
  3. Slice middle zero rows 2:8 from input or create them cheaply.
  4. Concat `[top2, middle6, reversed_top2]` along height.
- If number of filled top rows can vary, detect nonzero rows cheaply using ReduceMax over width/channel and place mirrored nonzero rows at bottom, but first test whether train/test always use exactly two filled rows. If fixed, hardcode it for better score.

Important:
- Graph output must be exactly `[1,10,30,30]` if Kaggle wrapper requires padded output, but internal computation should stay compact 10x5 and pad only at the final node.
- Input/output outer tensors are free; internal tensors are expensive.
- Minimize initializer scalar count.
- After generating ONNX, run shape inference and ORT profiler, report:
  - internal memory
  - params
  - cost
  - NeuroGolf score
- Try multiple variants and keep only the lowest cost valid model:
  A) Slice + Concat + final Pad
  B) Gather rows `[0,1,2,3,4,5,6,7,1,0]` if middle rows are already zero
  C) Pure ConstantOfShape/Where variant only if cheaper

Expected best model:
- likely Gather or Slice+Concat.
- Since rows 2..7 are already zero, the cheapest version may simply gather rows:
  output rows = input rows [0,1,2,3,4,5,6,7,1,0]
- Then final pad/cast to required output shape if needed.

Nejlepší sázka: Gather s indexy [0,1,2,3,4,5,6,7,1,0], protože prostřední řádky už jsou nuly a nemusíš vytvářet žádný nový middle tensor.

----------------------------------------------

tohle je task 114 na vstupu je maximalne 3x3 ktery je podle barev usporadan na vystup. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create the most score-optimized ONNX model for NeuroGolf ARC task114.

Task rule:
Input grid is tiny, max 3x3. Output is an expanded grid where each input cell becomes a connected 2-cell domino-like piece, arranged by color in a fixed “weaving” expansion.

Observed mappings:
- 2x2 input -> 4x4 output
- 3x2 input -> 5x4 output
- 3x3 input -> 5x5 output

Expansion rule:
For every non-black input cell at position (r,c), copy its color into output at:
  (r+c, 2*c)
  (r+c+1, 2*c+1)

This creates diagonal/right-down dominoes.
Black input cells still produce black/no cells.
The output canvas size is:
  H_out = H_in + W_in
  W_out = 2 * W_in
Then crop/remove the unused last row or column when needed to match the expected ARC output:
  for 2x2: 4x4
  for 3x2: 5x4
  for 3x3: 5x5

Implementation requirements:
- Build a pure ONNX graph, no Python runtime logic.
- Optimize for NeuroGolf score, not readability.
- Input is one-hot ARC tensor `[1,10,30,30]`, but the meaningful grid is only top-left HxW, with H,W <= 3.
- Output must be `[1,10,30,30]`; construct only the minimal internal useful tensor and pad at the very end.
- Avoid large internal `[1,10,30,30]` tensors.
- Use only cheap ops: Slice, Gather, Concat, Pad, Where/Mul/Add if needed.
- Prefer hardcoded extraction of the top-left 3x3 and direct placement into the small output.
- Minimize initializers/Constants because params count against score.
- Constants should be scalar/int64 shape helpers only where unavoidable.
- Do not create one-hot conversions if input is already one-hot.
- Preserve all 10 channels.

Target construction:
1. Slice input to small `[1,10,3,3]` or smaller if dynamic shape is known.
2. For each source cell `(r,c)`, extract `[1,10,1,1]`.
3. Place it into output coordinates:
   y0 = r+c, x0 = 2*c
   y1 = r+c+1, x1 = 2*c+1
4. Merge placements by addition/max. There should be no color conflict.
5. Build only the small final canvas `[1,10,5,5]` or `[1,10,5,4]` / `[1,10,4,4]` depending on task input size if the evaluator fixes input size. If dynamic H/W handling is expensive, specialize to the actual task shape distribution that gives best score.
6. Pad small canvas to `[1,10,30,30]` only as final graph output.

Try multiple versions and report measured NeuroGolf score:
A) generic max-3x3 model producing 5x5 then padded
B) shape-specialized variants if test set shape is fixed or detectable cheaply
C) ultra-compact version using Concat row construction instead of scatter-like placement

Important:
- File size does not matter except through params and memory.
- Do not use Resize/Tile to large tensors.
- Do not use full-grid internal masks.
- Use ONNX shape inference + ORT profiling to verify internal memory.
- The best solution is likely a hardcoded small tensor rearrangement with final Pad.

---------------------------------------------------

tohle je task 115 na vstupu jsou postupne barvy vertikalne nebo horizontalne a na vystupu jsou ty barvy v jendom sloupci nemaji presnou prechodovou hranu. Dej mi prompt pro onnx s nejlepsim moznym score

Zkopíruj do Codex/Cursoru:

Create a maximally score-optimized ONNX model for NeuroGolf ARC task115.

Task rule:
Input is a small one-hot ARC grid with several solid color regions arranged as sequential bands, either left-to-right vertical bands or top-to-bottom horizontal bands. The boundaries between colors are noisy / jagged, not perfectly straight.

Output is a compact canonical strip containing only the ordered colors:
- If input bands are vertical: output shape is 1 x N, one row, colors ordered left-to-right.
- If input bands are horizontal: output shape is N x 1, one column, colors ordered top-to-bottom.
- N is the number of distinct non-background colors appearing as bands.
- Each output cell/block uses one representative color from each band.
- Ignore jagged transition pixels; infer order by global position of each color, not by exact boundary.

Examples:
- 14x16 input with vertical noisy bands olive, red, cyan -> output 1x3: [olive, red, cyan].
- 9x7 input with horizontal noisy bands red, cyan, gray -> output 3x1: [[red], [cyan], [gray]].

Implementation target:
Build a tiny ONNX graph, no neural net, no training, pure tensor logic. Optimize for NeuroGolf score:
- Minimize internal tensor memory and initializer params.
- Avoid large intermediate tensors.
- Use int64/int32/bool where possible.
- Do not care about ONNX file size unless it reduces params or memory.
- Input/output 30x30 Kaggle padding is free only for graph I/O; internal tensors are expensive.

Suggested algorithm:
1. Crop/extract only the active grid area from the one-hot input.
2. Convert one-hot to class IDs if cheaper.
3. Compute for each color c=1..9:
   - presence mask
   - count per row and count per column
   - total count
   - min/max row and min/max column or weighted mean row/col.
4. Determine orientation:
   - vertical bands if colors have stronger separation by x/columns than by y/rows.
   - horizontal bands if stronger separation by y/rows.
   Practical cheap heuristic:
     range/variance of color centroid columns > range/variance of centroid rows => vertical, else horizontal.
5. Keep only present non-background colors.
6. Sort present colors by centroid column for vertical orientation, or centroid row for horizontal orientation.
7. Emit ordered color IDs as a 1xN or Nx1 compact output.
8. Pad only at the very final output to required [1,10,30,30] one-hot format.

Optimization requirements:
- Prefer small vectors over full HxW tensors after the initial crop.
- Avoid TopK/Sort if possible: since colors are only 1..9, produce order by repeated comparisons of centroid coordinates or by bucket scanning.
- If maximum number of bands in task is known from train examples as 3, specialize hard for N=3.
- Hardcode expected max input sizes if it reduces memory.
- Use constants only when they reduce score cost.
- Profile with ONNX Runtime shape inference/profiler and report:
  memory, params, cost = memory + params, score = max(1.0, 25.0 - ln(cost)).
- Try at least 3 graph variants and keep the lowest cost:
  A) centroid-based orientation/order
  B) row/column projection majority-based order
  C) specialized 3-color version with no dynamic sorting.

Return:
- Python script that builds `task115.onnx`
- validation code on synthetic examples matching the screenshots
- printed NeuroGolf score components
- short explanation of why this is near-minimal

Největší šance na top score: specializovat na max 3 barvy a nedělat obecný sort. Stačí spočítat centroid každé přítomné barvy a ručně seřadit 3 hodnoty přes Less/Where.


-------------------------------------------------

tohle je task 116 na vstupu je patern ktery je na vystupu dvakrat, jednou otoceny. Dej mi prmompt pro onnx s nejlepsim moznym score.

Použij tenhle prompt pro Codex/Cursor:

Create a minimal-score ONNX solver for ARC/NeuroGolf task116.

Task:
Input is always 3x4.
Output is always 6x4.
The output is made by stacking the original 3x4 input pattern and the same pattern rotated/flipped by 180 degrees.

Transformation:
Given input X with shape [3,4]:

top half of output:
Y[0:3, 0:4] = X

bottom half of output:
Y[3:6, 0:4] = rot180(X)
                 = X[::-1, ::-1]

So:
Y = concat_rows(X, reverse_rows_and_columns(X))

Example:
input row0 -> output row0
input row1 -> output row1
input row2 -> output row2
reverse(input row2) -> output row3
reverse(input row1) -> output row4
reverse(input row0) -> output row5

ONNX optimization goal:
Maximize NeuroGolf score, not readability.
Avoid huge internal tensors.
Do not pad to 30x30 until the final output if Kaggle wrapper requires [1,10,30,30].
Keep all internal tensors as small as possible, ideally only [1,1,3,4] and [1,1,6,4].
Use int/uint operations if possible, but choose whatever gives lowest cost.

Implementation idea:
1. Slice/extract the 3x4 active grid from input.
2. Create rot180 using Slice with negative steps on H and W axes.
3. Concat original and rot180 along height axis.
4. If required, final-pad/scatter into [1,10,30,30] only at the last node.

Measure actual NeuroGolf cost after ONNX shape inference and ORT profiling.
Try alternative implementations:
- Slice negative steps + Concat
- Gather rows/cols with constant indices [2,1,0] and [3,2,1,0]
- Precomputed index GatherND if it reduces params/memory

Pick the model with lowest memory + params, not smallest file size.

-----------------------------------------

tohle je task 117 na vstupu je jeden patern diagonalne k x ten je skopirovan otoceny zrcadlove dokola paternu x. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Task117 NeuroGolf/ARC ONNX optimalizace.

Cíl: vytvoř co nejlevnější validní ONNX model pro transformaci 14x14 → 14x14.

Popis úlohy:
- Vstup je vždy 14x14.
- Na vstupu jsou dva objekty:
  1. barevný šikmý/diagonální pattern jedné barvy,
  2. objekt ve tvaru X jiné barvy.
- Výstup zachová X beze změny.
- Šikmý pattern se má zkopírovat kolem X ve všech zrcadlových/rotačních polohách.
- Prakticky: vezmi původní pattern relativně k centru X a vytvoř jeho symetrické kopie podle os X:
  - originál,
  - zrcadlení vlevo/vpravo,
  - zrcadlení nahoře/dole,
  - kombinace obou zrcadlení.
- Výsledkem je „věnec“ stejného patternu okolo X.
- Barva kopírovaného patternu zůstává stejná jako ve vstupu.
- Barva X zůstává stejná jako ve vstupu.
- Pozadí zůstává 0.

Optimalizační požadavky:
- Maximalizuj NeuroGolf score, ne čitelnost.
- Nepoužívej Conv, velké Constant tensory ani velké interní feature mapy.
- Vstup/výstup `[1,10,30,30]` jsou Kaggle formát, ale reálná mřížka je 14x14.
- Velký output/input je zdarma, ale interní tensory jsou drahé.
- Interně pracuj jen s cropem 14x14 nebo menším.
- Používej levné ONNX operace: Slice, Gather, ReduceSum/ReduceMax, Equal, Where, Mul, Add, Cast, NonZero/ArgMax pokud vychází levně.
- Ideálně vytvoř masku X barvy a masku pattern barvy.
- Najdi bounding box / centrum X podle ne-background barvy, která tvoří X.
- Pattern masku transformuj čistě indexově:
  - pro každý pixel patternu spočti relativní souřadnice vůči centru X,
  - vytvoř pozice `(cx ± dx, cy ± dy)`,
  - zapiš tam stejnou barvu patternu.
- X masku vlož zpět přes pattern pomocí Where, aby X nebyl přepsán.
- Nakonec výsledek padni zpět do `[1,10,30,30]`.

Důležité:
- Pokud je jednodušší hardcodovat 14x14 index-grid konstanty, udělej to, ale minimalizuj počet parametrů.
- Preferuj generování souřadnic přes Range/meshgrid-like logiku, pokud je levnější než konstantní 14x14 mapy.
- Otestuj několik variant a reportuj memory, params, cost, score.
- Vyber model s nejnižším `memory + params`, ne nejmenší `.onnx` soubor.

-------------------------------------------------------------------------------------------------

tohle je task 118 na vstupu ma nahodne sede pozadi s cervenymi nedokoncenymi krizi. Na vystupu jsou doplneny svetle modrou do krize. Dej mi prmopt pro vytvoreni onnx s nejlepsim moznym score
Create the smallest/highest-score ONNX solver for NeuroGolf ARC task118.

Task rule:
- Input/output are always 20x22.
- Background is random gray/black noise and must be preserved exactly.
- Red pixels form one or more incomplete plus/cross shapes.
- Output keeps all original pixels unchanged and only fills the missing arms of each red cross with light-blue/cyan pixels.
- Red = original given cross pixels.
- Cyan = inferred missing pixels.
- The completed shape is a 5-cell plus centered on the red structure:
  center plus its 4-neighborhood arms: up, down, left, right.
- Existing red pixels remain red; only missing cells of the same plus are written as cyan.
- Ignore gray/black background patterns completely except as pixels to preserve.

Implementation target:
- Build ONNX directly with onnx.helper, not PyTorch export.
- Optimize for NeuroGolf score: minimize internal tensor memory + params.
- Input one-hot tensor is likely [1,10,30,30] or [1,10,H,W]; crop/use only valid 20x22 region.
- Output must be Kaggle-required [1,10,30,30], with solved 20x22 region placed top-left and padded elsewhere as zeros/background.

Preferred ONNX logic:
1. Extract class/color plane for red.
2. Compute candidate cross centers from red pixels using cheap neighbor logic:
   - A center is any red pixel that has enough red support in vertical/horizontal arms, or any empty cell that is between red arm fragments.
   - More robust: use 4-neighborhood/2-step patterns around red to infer the unique plus center.
3. For each inferred center, create a 5-cell plus mask using shifts/Pad/Slice/Concat or a tiny Conv if cheaper.
4. `missing = plus_mask AND NOT red_mask`.
5. Write cyan on `missing`.
6. Preserve all other input colors exactly:
   `out = input_onehot`, then replace only missing cells with cyan one-hot.

Score constraints:
- Avoid full [1,10,30,30] intermediates whenever possible; operate on [1,1,20,22] masks.
- Avoid large constants. Use scalar constants only.
- Prefer Slice/Gather/Concat/Pad/Where/Or/And/Cast over Conv if Conv weights increase params too much.
- Measure with the NeuroGolf scorer after every attempt. Report memory, params, cost, score.
- Try at least these variants:
  A) neighbor-shift boolean implementation,
  B) tiny convolution kernel implementation for red-neighbor counts,
  C) hardcoded crop/pad minimal graph.
- Keep the best-scoring valid ONNX only.

Validation:
- Must exactly match all train examples.
- Test with random gray background and multiple incomplete red crosses.
- Ensure existing red pixels are never recolored cyan.
- Ensure cyan is added only where the completed plus requires missing cells.

---------------------------------------------------

tohle je task 119 na vstupu je cervena stena z jedna strany a diagonalni svetle modre bloky z kterych pokracuej diagonalne zelena ktera se od cervene steny odrazy o 90 stupnu az do kraje. Dej mi prompt pro vytvoreni onnx s nejlspim moznym score

Použij tenhle prompt pro Codex/Cursor:

We need solve NeuroGolf ARC task119 with a minimal-score ONNX generator.

Task rule:
Input/output are fixed 12x12.
There is a solid red vertical wall on one side, usually the rightmost column(s). There are one or more cyan seed cells on a diagonal. Output keeps the input unchanged and adds a green diagonal ray starting from the cyan diagonal, continuing with slope -1/+1 toward the red wall. When the ray reaches the red wall side, it reflects by 90 degrees and continues diagonally to the grid edge. Red wall cells remain red, cyan cells remain cyan, added path cells are green.

Goal:
Create a Python script that writes a valid ONNX model with the best NeuroGolf score, minimizing internal tensor memory + initializer params. Do not optimize for file size unless it reduces cost.

Implementation strategy:
- Fixed input shape [1,10,12,12], output [1,10,12,12].
- Use channel 0 as color-index grid if this repo convention stores ARC colors in one-hot/channel format; inspect existing task scripts and match their format exactly.
- Preserve all nonzero input cells.
- Detect red wall as color 2 cells.
- Detect cyan seeds as color 8 cells.
- Build candidate reflected diagonal path using cheap coordinate masks for 12x12:
  - cells on the same / diagonal as cyan seeds,
  - then reflected from the red wall with 90-degree turn,
  - stop at borders,
  - never overwrite red or cyan/nonzero cells.
- Paint added cells green color 3.
- Prefer constant coordinate grids as small int64/float tensors only if cheaper than many Slice/Concat ops.
- Avoid creating full [1,10,12,12] intermediate tensors repeatedly. Work mostly on [12,12] or [1,1,12,12] masks, then one-hot/where only at final composition.
- Keep graph static and branchless: Equal, ReduceMax, Where, Add/Sub, Mul, Less/Greater, And/Or, Cast.
- After ONNX shape inference and ORT profiling, print:
  params count,
  internal tensor memory,
  total cost,
  NeuroGolf score.
- Try multiple variants and keep the lowest measured cost:
  1. full coordinate-mask formulation,
  2. precomputed lookup masks for each possible cyan diagonal / wall side,
  3. hardcoded per-cell formulas for 12x12 if params stay lower.
- Validate on all train pairs and hidden-style variants.

Important:
Do not use a large lookup table of full outputs unless measured cost is lower.
Do not pad to 30x30 if this task expects 12x12 output.
Return exactly [1,10,12,12].

Největší riziko: přesná geometrie odrazu. Nech Codex nejdřív extrahovat train páry a ověřit, jestli zelená cesta je opravdu anti-diagonal + reflected anti-diagonal, nebo jestli se počítá z nejbližšího cyan bloku ke stěně.

---------------------------------------------

tohle je task 120 na vstupu jsou obdelniky ruzne barvi jejichz vnitrek je vyplnen svetle modrou. Dej mi prompt pro vytvoreni onnx s nejlespim moznym score.
Create the most score-optimized NeuroGolf ONNX solver for ARC task120.

Task:
Input/output size is fixed 12x14. The input contains several solid colored rectangles of different colors on black background. Output keeps all rectangles in the same positions and replaces only the true interior cells of each rectangle with light blue/cyan. Rectangle borders remain their original color. Black background stays black.

Observed rule:
For every non-black connected rectangular object:
- detect its bounding box / occupied rectangle
- keep border pixels unchanged
- set pixels strictly inside the rectangle to color 8
- only fill if the cell is inside an existing colored rectangle, not background
- handles rectangles of different sizes and colors
- examples include blue, green, red rectangles, but code must be color-agnostic for nonzero colors

Important:
Optimize for NeuroGolf score, not file size.
Use pure ONNX graph construction, no ML training.
Minimize params and internal tensor memory.
Prefer compact boolean masks and simple Slice/Pad/Where/And/Or ops.
Input/output tensors must satisfy Kaggle shape, but internal tensors should remain as small as possible.

Implementation strategy:
Input is one-hot [1,10,30,30], but active grid is top-left 12x14.
Slice active area to [1,10,12,14].
Convert to color grid or nonzero mask:
- nz = any channel 1..9 over color axis, shape [1,1,12,14]
- color output initially equals active input.

Interior condition for a cell:
A cell is interior if it is nonzero and has nonzero neighbors up, down, left, right.
This works for filled rectangles: border lacks at least one 4-neighbor inside object, interior has all four.
Compute:
up    = nz shifted down / compare cell above
down  = nz shifted up
left  = nz shifted right
right = nz shifted left
interior = nz & up & down & left & right

Then output:
- for color channels 0..9:
  - channel 8 = where(interior, 1, original channel 8)
  - all other channels = where(interior, 0, original channel)
This recolors only interior pixels to light blue while preserving borders and background.

Score optimization:
Try multiple ONNX variants and measure NeuroGolf score:
1. Boolean neighbor-mask version using Slice + Pad + And.
2. Conv-like 3x3/4-neighbor sum version only if it reduces memory/params.
3. Direct channel-wise Where version without ArgMax if one-hot is already available.
Pick the lowest cost model.

Avoid:
- ArgMax/OneHot roundtrips if unnecessary.
- Large [1,10,30,30] internal tensors before final padding.
- full 30x30 computations.
- constants larger than scalar/small shape constants.

Final output:
Pad the [1,10,12,14] result back to required [1,10,30,30].
Save as task120.onnx.
Run checker/shape inference/profiler and print memory, params, cost, score.
Return the best measured variant only.

------------------------------------------

tohle je task 121. vstup je vzdy 13x13 na vstpu jsou paterny jeden z nich ma v sobe svetle modrou kostku tenhle patern je output s tim ze se svetle modra kostka nahradi za barvu patternu. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create a maximally score-optimized ONNX solver for ARC/NeuroGolf task121.

Task:
- Input is always 13x13.
- Output is 3x3.
- There are several disconnected 3x3 cross/plus-like patterns on black background.
- Exactly one pattern contains a light-blue marker cell, color 8.
- Select that marked pattern.
- Output the selected 3x3 pattern’s bounding box, but replace the light-blue marker cell with the main color of that selected pattern.
- All other input patterns must be ignored.

Important example behavior:
- If the marked object is a 3x3 plus made mostly from color C and has one cell value 8 inside it, output the same 3x3 plus, with the 8 cell changed to C.
- Background remains 0.
- Output size is exactly 3x3.

Optimization goal:
- Minimize NeuroGolf cost = internal tensor memory + params.
- Do not optimize ONNX file size unless it reduces params or internal tensors.
- Prefer scalar/small-tensor logic.
- Avoid full one-hot `[1,10,13,13]` tensors unless absolutely necessary.
- Avoid large intermediate tensors, Conv, MatMul, or big Constant tables.
- Use int/uint/bool operations where possible.
- Graph input/output tensors are free, but all internal tensors are counted.

Suggested ONNX strategy:
1. Input shape likely `[1,10,13,13]` one-hot ARC format.
2. Extract class planes only as needed:
   - plane8 = input[:,8,:,:] as marker mask.
   - nonzero/color planes only if needed for reconstructing values.
3. Locate the marker:
   - Find coordinates `(r8, c8)` where color 8 is present.
   - Because there is exactly one marker, use `NonZero`, `Gather`, or equivalent minimal coordinate extraction.
4. The selected 3x3 object is the 3x3 window centered around / containing the marker’s object.
   - Since all candidate objects are 3x3 patterns, determine the object bounding box as rows `r0..r0+2`, cols `c0..c0+2`.
   - Robust way: use local neighborhood around marker and/or find nearby nonzero cells connected to marker within Chebyshev distance <= 1, then infer min row/col.
   - In this task the marker is inside the selected 3x3 pattern, so `r0 = min row among nonzero cells within 3x3 around marker`, `c0 = min col among nonzero cells within 3x3 around marker`.
   - Simpler if examples guarantee marker is center: `r0 = r8 - 1`, `c0 = c8 - 1`. But prefer robust min-neighbor if cost is still low.
5. Crop selected 3x3 from input/value grid:
   - Convert one-hot input to value grid `[13,13]` only if cheaper than processing planes.
   - Crop `value[r0:r0+3, c0:c0+3]`.
6. Determine replacement color C:
   - C is the nonzero, non-8 color in the selected 3x3 crop.
   - Because the pattern is monochrome except marker, compute C from any crop cell where value != 0 and value != 8.
   - Use max/reduce over masked crop: `C = ReduceMax(Where((crop != 0) & (crop != 8), crop, 0))`.
7. Replace marker:
   - `out = Where(crop == 8, C, crop)`.
8. Convert output back to required format if needed:
   - If competition expects one-hot `[1,10,30,30]`, emit 3x3 result padded into top-left of `[1,10,30,30]` only at the final step.
   - If task harness expects raw `[3,3]`/`[1,1,3,3]`, output directly.
   - Keep padding/final one-hot construction as late as possible because graph outputs are free but internal padded tensors are expensive.

Implementation requirements:
- Generate a valid ONNX model and immediately test it with onnxruntime on all available train examples.
- Print:
  - inferred input/output shapes,
  - params count,
  - internal memory estimate from ORT profiling / shape inference,
  - NeuroGolf score.
- Try multiple variants and keep the best scoring valid model:
  A. Simple marker-centered crop variant.
  B. Robust local-min bounding-box variant.
  C. Plane-based variant avoiding full value-grid if lower memory.
- Save the best model as `task121_best.onnx`.

Do not assume opset 10. Use whichever opset gives the lowest measured NeuroGolf score while staying valid.
Prioritize correctness first, then aggressively reduce internal tensor memory and params.

------------------------------------

tohle je task 122. Na vstupu je prerusovany zeleny radek nebo sloupec s cervenym ctvercem ktery je pozunut o dva bloky dolu pokuz je zeleny sloupec a o dva bloky doprava jeli zeleny radek. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create a maximally score-optimized ONNX model for NeuroGolf ARC task122.

Task rule:
- Input and output are fixed shape 7x13, one-hot ARC tensor format.
- Background is black 0.
- There is a dashed green line, either horizontal or vertical.
- There is a red square/block offset from the dashed green line:
  - if the green object is a horizontal dashed row, copy/paint the red square pattern two columns to the right onto the green row
  - if the green object is a vertical dashed column, copy/paint the red square pattern two rows down onto the green column
- Output keeps the original green dashed line and adds/repositions the red cells onto the green line-aligned target location.
- Do not solve visually by hardcoding this sample only; implement the general rule for 7x13 inputs.

Optimization goal:
- Minimize NeuroGolf cost = internal tensor memory + initializer params.
- Prefer zero or near-zero initializers.
- Avoid Conv, MatMul, large Constant tensors, one-hot expansion beyond what is already required, and large intermediate tensors.
- Use only cheap ONNX ops: Slice, Gather, ReduceSum/ReduceMax, ArgMax if useful, Equal/Greater, Where, Cast, Add/Mul, Unsqueeze/Squeeze, Concat only if needed.
- Keep all intermediate tensors as small as possible; prefer boolean/uint8/int64 masks over float tensors when valid.
- Final output must be padded/formed only at the end if Kaggle wrapper requires [1,10,30,30]; graph inputs/outputs are free but internal tensors count.

Implementation strategy:
1. Extract the active 7x13 grid from channel format.
2. Build green mask `G = input == 3` and red mask `R = input == 2`.
3. Detect orientation cheaply:
   - horizontal if any row has multiple green cells.
   - vertical if any column has multiple green cells.
4. For horizontal case:
   - shift red mask by +2 columns.
   - target red mask = shifted red mask AND row-support of green dashed line, or equivalently shifted red aligned to the green row.
5. For vertical case:
   - shift red mask by +2 rows.
   - target red mask = shifted red mask AND column-support of green dashed line.
6. Combine:
   - keep original input colors
   - set target red cells to color 2
   - preserve green cells not overwritten unless task examples require red replacing green at overlap.
7. Output exact 7x13 grid, then pack/pad to required competition output.

Important:
- Generate a Python script that builds the ONNX directly with `onnx.helper`.
- Hardcode fixed dimensions 7 and 13 for lower memory.
- After building, run ONNX shape inference and ORT test on several synthetic cases:
  - horizontal green dashed row with red block two columns left of target
  - vertical green dashed column with red block two rows above target
  - edge-safe cases where shifted red would leave grid
- Print NeuroGolf-relevant estimates: initializer scalar count, inferred intermediate tensor shapes, and expected memory.
- Try at least two graph variants and keep the lower-cost ONNX:
  A) shift red mask using Slice + Pad/Concat
  B) construct target positions from reduced row/column masks with broadcasting
- Save the best model as `task122_best.onnx`.

------------------------------------------

tohle je task 123. Na vstupu je pattern ktery je opakovan tak aby vystup byl 10x10. Dej mi prompt pro vytvoreni onnx s nejllepsim moznym score
Create a maximally score-optimized ONNX solver for NeuroGolf ARC task123.

Task:
- Input is always a 5x5 grid.
- Output must be 10x10.
- The visible 5x5 pattern is repeated/tiled 2x2 to fill the 10x10 output.
- No reasoning about colors is needed: every output cell is simply:
  output[r, c] = input[r % 5, c % 5]

Implementation goal:
- Build the smallest/cheapest ONNX possible.
- Minimize NeuroGolf cost = internal tensor memory + initializer params.
- Avoid large intermediate tensors except the required final output.
- Graph input/output tensors are free, but internal tensors are expensive.
- Prefer pure ONNX ops with no unnecessary constants.
- Do not optimize for file size unless it also reduces params or internal memory.

Recommended ONNX strategy:
1. Input shape is `[1, 10, 5, 5]` one-hot/color-channel ARC tensor.
2. Extract only the active 5x5 spatial region if needed.
3. Tile spatially by 2 in height and 2 in width to produce `[1, 10, 10, 10]`.
4. Output exactly `[1, 10, 10, 10]`.

Use the cheapest measured construction among:
- `Tile` with repeats `[1,1,2,2]`
- or `Concat` input with itself along W, then Concat result with itself along H.

Test both versions and keep the lower NeuroGolf score-cost version. Concat may be better if it avoids repeat initializer overhead, but measure.

Important:
- Do NOT create masks.
- Do NOT use color comparisons.
- Do NOT use loops.
- Do NOT generate coordinate grids.
- Do NOT pad from larger intermediates.
- Do NOT materialize anything bigger than the final 10x10 unless unavoidable.

Validation:
For all colors/channels and all positions:
`Y[:, :, r, c] == X[:, :, r % 5, c % 5]`

Save as task123.onnx and print:
- params count
- internal tensor memory
- total cost
- NeuroGolf score

------------------------------------------------

tohle je task 124. na vstupu dostane patern ktery je zvetsen aby byl 10x10. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create an ultra-optimized ONNX model for NeuroGolf ARC task124.

Task rule:
- Input is a small colored pattern on black background.
- Output must always be 10x10.
- The non-black pattern from input is scaled/expanded to fit a 10x10 canvas.
- Preserve the original foreground color.
- Black stays black.
- Examples:
  - 6x10 input diagonal/step blue pattern -> 10x10 output, vertically expanded to 10 rows.
  - 8x10 input red vertical/side pattern -> 10x10 output, vertically expanded to 10 rows.
- Input width may already be 10; height can be smaller than 10.
- Treat this mainly as nearest-neighbor resize of the whole input grid to 10x10, not as object detection.

Optimization target:
- Maximize NeuroGolf score, not readability.
- Minimize cost = internal tensor memory + initializer scalar params.
- Graph input/output tensors are free; internal tensors are expensive.
- Avoid large float tensors and avoid unnecessary constants.
- Prefer uint8/bool/int64 logic where possible.
- Avoid Conv/Gemm/large one-hot expansions.
- Avoid full `[1,10,30,30]` intermediates if final Kaggle wrapper needs padding; compute compact 10x10 first, then pad only at the very end.
- Use only tiny index constants if needed.

Implementation strategy:
1. Extract the active input grid from the provided `[1,10,30,30]` style ARC tensor if that is the project format:
   - Use only the true task area.
   - For this task the useful input is Hx10 where H is 6 or 8 in shown examples.
2. Produce a 10x10 output by nearest-neighbor row resampling:
   - output_row r samples input_row floor(r * H / 10)
   - output_col c samples input_col floor(c * W / 10)
   - Since W is 10 in examples, columns can usually pass through unchanged.
3. Preserve all cell values exactly.
4. If dynamic H is hard in ONNX, specialize with cheap branch/detection:
   - detect whether rows 6..9 are empty to distinguish 6x10 vs 8x10 or other heights.
   - Use precomputed Gather row index constants:
     - for H=6: [0,0,1,1,2,3,3,4,4,5]
     - for H=8: [0,0,1,2,3,4,4,5,6,7]
   - Select correct gathered result with Where.
5. Columns:
   - If width is always 10, do not resample columns.
   - If variable width appears, apply same nearest-neighbor Gather on columns with tiny constants.

Deliverables:
- A single Python script that builds and saves `task124.onnx`.
- Include a small local verifier with the shown examples encoded as numpy arrays.
- Print NeuroGolf-relevant stats:
  - initializer scalar count
  - inferred internal tensor shapes
  - estimated internal memory
- Try several graph variants and keep the smallest measured NeuroGolf cost:
  A) direct Gather rows only
  B) Slice + Gather
  C) Where-selected H=6/H=8 branches
  D) hardcoded 6/8 if benchmark data confirms only one height per hidden split is impossible
- Do not optimize ONNX file size unless it also reduces params or internal memory.
- Do not force a specific opset unless it measurably improves score.

Important:
The best solution is probably a tiny Gather-based nearest-neighbor resize to 10x10. Any solution using image resize ops, convolution, or one-hot expansion is likely trash for NeuroGolf because it creates unnecessary internal memory/params.

---------------------------------------------------

tohle je task 125 na vstupu dostane ruzove obdelniky na svetle modrem pozadi ktere jso uvyplneny zlutou maji li nejakou mezeru a ohraniceni zelenou. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt pro Codex/Cursor:

We need solve NeuroGolf ARC task125 with a maximally score-optimized ONNX model.

Task rule:
Input/output are 15x15 grids, one-hot encoded as [1,10,15,15].
Background is light blue/cyan.
Input contains magenta/pink rectangle-like shapes, sometimes hollow / with missing interior space.
Output keeps the magenta shapes, fills enclosed holes/interior gaps with yellow, and adds a 1-cell thick green border around each detected magenta rectangle/object.

Important examples:
- A solid magenta rectangle becomes same magenta rectangle surrounded by green 1-cell frame.
- A hollow magenta rectangle keeps magenta outline, its enclosed empty interior becomes yellow, and the outside around the whole rectangle becomes green.
- Small partial magenta blocks also get green border around their bounding box/object; only enclosed inside/background cells within the magenta rectangle become yellow.
- Background remains unchanged elsewhere.

Goal:
Create the smallest/lowest-cost ONNX possible, not pretty code. Optimize NeuroGolf score, not file size.

Implementation strategy:
1. Extract class planes:
   - bg/cyan plane
   - magenta plane
2. Detect magenta connected components / rectangle bounding boxes.
3. For every magenta object, compute its bounding box:
   rows containing magenta, cols containing magenta.
4. Create green mask:
   cells in bbox dilated by 1 in 4-neighborhood/rectangle-frame sense, excluding original magenta and excluding yellow-fill cells.
   Simpler: for each component bbox, set rows [rmin-1:rmax+2], cols [cmin-1:cmax+2] as green frame, then overwrite original magenta and yellow after.
5. Create yellow mask:
   inside bbox cells that are background and enclosed by the magenta rectangle/object.
   For this task, interior holes are background cells inside the component bbox not occupied by magenta and not on the external green border.
6. Final priority:
   background default cyan
   green border
   yellow interior fill
   magenta original pixels highest priority if overlap.

Optimization hints:
- Input is fixed 15x15, output 15x15.
- Avoid large internal tensors; final output [1,10,15,15] is free as graph output, but intermediates are counted.
- Prefer bool/int masks with small spatial tensors.
- Avoid Conv if it creates large float intermediates unless measured better.
- Use Slice/Gather/ReduceMax/Where/Equal/Greater/Cast/Concat/Unsqueeze/Expand only if they reduce memory/params.
- Constants are params, so keep constant tensors tiny. Prefer scalar constants and shape tricks.
- Hardcode 15x15 logic if it lowers cost.
- Do not care about generality beyond this exact task.
- Test with ONNX Runtime and NeuroGolf scorer after every variant.
- Try multiple variants:
  A) bbox-based using row/col ReduceMax over magenta mask.
  B) morphology-based dilation of magenta mask for green, and flood-fill/inversion-style hole detection if bbox version fails.
  C) brute-force 15x15 coordinate constants only if params remain low enough.

Expected output colors:
- cyan background unchanged
- magenta original rectangles unchanged
- yellow fills holes/interior gaps
- green surrounds each rectangle/object by one cell.

Deliver:
- a Python script that builds the ONNX directly with onnx.helper
- no argparse, hardcoded output path
- print memory, params, cost, score if scorer available
- save as task125.onnx

--------------------------------

tohle je task 126 na vstupu jsou obdelniky s jednim chybejicim blokem uprostred a ve sloupci kde tento blok bychi  je na spodnim radku umisten zluty blok. Dej mi prmpt pro vytvoreni onnx s nejlepsim moznym score.
Task 126 NeuroGolf / ARC-AGI ONNX optimization prompt:

Create the smallest/highest-score ONNX model for this exact 8x8 -> 8x8 transformation.

Input:
- Shape is always 8x8, one-hot or class-grid depending on existing NeuroGolf scaffold.
- Background is 0.
- There are one or more magenta shapes.
- Each shape is a 3x3-ish hollow/defective rectangle pattern with exactly one missing internal block.
- The missing block position determines a column.
- Output keeps the input unchanged, but adds one yellow block on the bottom row in the same column as each missing block.
- Yellow color is 4.
- Magenta color is 6.

Rule:
For every magenta component, find the empty cell that is inside the component’s implied 3x3 bounding box / concavity. In that same x-coordinate, set output[7, x] = 4. Everything else is copied from input.

Optimization target:
- Maximize NeuroGolf score, not readability.
- Minimize internal tensor memory first, then params.
- Avoid large [1,10,8,8] intermediates if possible.
- Prefer integer/bool ops, Slice/Gather/Reduce/Where/Equal/And/Or.
- Do not pad or create 30x30 unless the evaluator wrapper absolutely requires it, and if required, pad only at the final output.
- Do not use Conv unless it demonstrably reduces cost.

Implementation idea:
1. Extract class grid or magenta channel only.
2. Build candidate mask for empty cells:
   empty = input == 0
   A missing center cell is detected when it has magenta support on both horizontal/vertical sides within the local 3x3 object neighborhood.
   Robust detector:
   candidate = empty AND (
       has magenta left/right/top/bottom around it in the component’s bounding 3x3 neighborhood
       OR it is the only zero hole/concavity inside a 3x3 box containing magenta pixels
   )
3. Reduce candidate mask over rows to get columns:
   cols[x] = any_y candidate[y,x]
4. Create bottom marker row:
   marker[7,x] = 4 where cols[x] is true, else 0.
5. Output = input except bottom-row marker cells are yellow.

For this task, a very cheap specialized solution is acceptable:
- Since output only adds yellow blocks at bottom row, directly compute yellow columns from zero cells surrounded by magenta structure.
- Then Scatter/Where yellow onto row 7.
- Copy all other cells unchanged.

Return:
- A Python script that builds and saves the ONNX.
- Include a local verifier on the shown examples.
- Print NeuroGolf-relevant stats: initializer scalar count, inferred internal tensor memory estimate, and output correctness.


-------------------------------------------

tohle je task 127, vstup je sedou rozdelen nanekolik casti kazda cast v sobe ma blok barvy ktera je namapovana na jinou barvu, tou je vyplnenea cela cast na vystupu. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tento prompt do Codex/Cursor:

We need solve NeuroGolf ARC task127 with a maximally score-optimized ONNX model.

Task:
Input/output are 3x11 grids. Background is dark teal/0. The input contains 3 horizontal rows separated by empty rows. In each active row, gray separator blocks split the row into several segments. Each segment contains exactly one colored marker block. The marker color maps to a different output fill color. On output, keep the same 3 active rows and replace each segment with the mapped fill color across the whole segment. Gray separator positions remain gray. Empty/background rows remain background.

Observed mapping from the example:
blue marker -> magenta fill
red marker -> orange fill
gray separators stay gray
background stays background

Example behavior:
Input active row:
[0,0,0,gray,0,0,0,gray,0,0,0]
with marker row variant:
[0,blue,0,gray,0,red,0,gray,0,blue,0]
Output:
[magenta,magenta,magenta,gray,orange,orange,orange,gray,magenta,magenta,magenta]

General rule:
For each row independently:
- Gray cells are barriers/separators and copied unchanged.
- The contiguous non-gray segment between barriers is filled with the mapped output color determined by the non-background marker inside that segment.
- Segment width can be inferred from gray barriers; do not hardcode only one row unless that gives lower NeuroGolf cost for all hidden tests.
- Output shape is exactly same as input, 3x11.

Optimization target:
Minimize NeuroGolf cost = internal tensor memory + parameters. Graph input/output tensors are free, but all intermediate tensors after ONNX shape inference/profiling are expensive. Prefer small integer/bool tensors, constants as few scalar initializers as possible, and direct slice/concat logic over large one-hot grids.

Implementation strategy:
Create a Python script that builds an ONNX model directly with onnx.helper, no PyTorch export.

Recommended compact solution for fixed 3x11:
1. Input is one-hot or ARC tensor according to existing NeuroGolf harness conventions. Follow the repo’s existing task builder format.
2. Extract the relevant 3x11 class/color plane or grid representation with minimal Slice/Gather ops.
3. Detect marker colors in each segment:
   - Segment 0: cols 0..2, marker can be anywhere inside.
   - Segment 1: cols 4..6.
   - Segment 2: cols 8..10.
   Gray separators at cols 3 and 7 are copied as gray.
4. For each segment and each active row, reduce marker presence over the segment:
   - has_blue = any(segment == blue)
   - has_red = any(segment == red)
   - fill_color = magenta if has_blue else orange if has_red else background
5. Broadcast/fill each 1x3 segment with its fill color.
6. Concatenate:
   seg0_fill, gray_col, seg1_fill, gray_col, seg2_fill
   for each of 3 rows.
7. Preserve empty rows/background if the harness actually gives larger padded tensors; otherwise output exactly 3x11.

Important:
- Try hardcoded fixed slices first because task dimensions are fixed and this should score best.
- Avoid full [1,10,30,30] internal tensors until the final output/pad required by the competition wrapper.
- If the repo requires `[1,10,30,30]` one-hot output, build compact `[1,10,3,11]` internally and only Pad/Slice/Concat to final shape at the very end.
- Use ArgMax/Equal/ReduceMax/Gather/Where only on tiny tensors.
- Prefer bool masks and uint8/int64 where accepted by ONNX Runtime.
- Benchmark memory/params after each variant with the official NeuroGolf evaluator.

Generate:
- `task127_build.py` that writes `task127.onnx`
- a quick verifier using the shown example and random marker placements inside the 3 segments
- print params, memory, cost, and score

Search for the lowest-cost variant among:
A) slice/reduce per segment
B) direct fixed column marker reads if marker location is stable in hidden tests
C) lookup-table mapping marker colors to output colors
D) pure one-hot channel construction only at the final step

Choose the ONNX with the best measured NeuroGolf score, not the prettiest graph.

-----------------------------------

tohle je task 128. nav stupu jsou na dole obdelniky ktere jsou posunuty nahoru o pocet bloku kolik maji na visku. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score.

Použij tento prompt pro Codex/Cursor:

We need create a minimal-score ONNX solver for NeuroGolf ARC task128.

Task rule:
Input and output are 15x15 one-hot ARC grids, shape likely [1,10,15,15].
The input contains several solid colored rectangles touching the bottom border.
Each rectangle must be moved upward by exactly its own height.
Horizontal position and color stay unchanged.
The original bottom rectangle cells become background 0.
Output shape is still 15x15.

Example:
- blue rectangle height 4 at bottom rows -> appears 4 rows higher
- red rectangle height 2 -> appears 2 rows higher
- yellow rectangle height 4 -> appears 4 rows higher

Goal:
Generate the ONNX with best NeuroGolf score, not readability. Minimize internal tensor memory + initializer params. Do not care about ONNX file size except when it affects params/memory.

Important optimization direction:
Because grid is fixed 15x15, avoid large full-grid intermediate tensors if possible.
Use channel/color preserving operations.
Prefer Slice + Pad + Where/Mul/Add only if memory is low.
Need detect bottom-connected colored columns/rectangles and shift each column segment upward by its height.

Robust algorithm:
1. Extract class grid or one-hot channels excluding background.
2. For each non-background color channel independently:
   - identify bottom-connected vertical run in each column.
   - Since rectangles are solid and touch bottom, height per column is number of consecutive colored cells from bottom upward.
   - For each possible height h = 1..15:
       mask columns where bottom run height == h for that color.
       shift those cells upward by h rows.
3. Clear original positions by constructing output from shifted masks only.
4. Preserve exact color channels.
5. Background channel can be generated as 1 - any_foreground if required by submission format.

Implementation hints:
- Fixed 15 rows allows hardcoded row slices.
- For each h, create shifted version:
  shifted_h[:, :, 0:15-h, :] = input[:, :, h:15, :]
  but keep only rectangles whose height is h.
- Detect height h cheaply from bottom rows:
  For a color channel c and column x:
    run_ge_k = AND of c at rows 15-k..14 for k cells.
    height == h means run_ge_h AND NOT(row 14-h is color), except h=15.
  Broadcast this column mask across rows.
- Since only bottom rectangles matter, no need for general connected components.

Try multiple ONNX variants and measure with official NeuroGolf scorer:
Variant A:
- one-hot channel implementation, loop over h=1..15, using ReduceMin/And over bottom slices, Not, Mul, Pad, Sum.
Variant B:
- class-index implementation using ArgMax, Equal(color), then same logic, reconstruct one-hot at end.
Variant C:
- hardcoded per-row formulas avoiding full broadcast where possible.

Score target:
Keep params almost zero. Avoid Constants with large tensors.
Avoid producing internal [1,10,15,15] repeatedly if possible; full tensor costs 9000 bytes each for float32, less for bool/int8 if supported by scorer/runtime.
Use bool masks where possible.
Use int64 only where mandatory; cast late.

Output:
Save model as task128.onnx.
Then run shape inference + ORT validation + NeuroGolf scoring.
Report:
- memory
- params
- cost
- score
- list of largest internal tensors
- exact ops used
- whether all training examples pass

Do not stop at first valid model. Iterate for lower cost.

Největší trik: neřešit komponenty obecně. Stačí detekovat výšku bottom-runu pro každý sloupec a posunout buňky o tu samou výšku nahoru.

-----------------------------------

tohle je task 129 na vstupu je vzdy 3x3 s bavama uvnitr te vystup je 3x3 vyplneny barvou ktere je na vstupu nejvic. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Task129: Build the smallest/highest-score ONNX solver.

Rule:
- Input is always 3x3 ARC grid, one-hot or scalar depending on existing NeuroGolf wrapper.
- Output is 3x3.
- Find the color that appears most often in the 3x3 input.
- Fill the entire 3x3 output with that majority color.

Implement aggressively optimized ONNX:
1. Avoid any large internal tensors. Output is only 3x3, so no padding/30x30 unless the evaluator wrapper requires it.
2. Count occurrences of each color 0..9 over the 9 input cells.
3. Take ArgMax over the 10 counts to get majority color.
4. Broadcast/tile that single color to 3x3.
5. If using one-hot format:
   - reduce input over H,W → counts [10]
   - ArgMax → color id
   - OneHot/color reconstruction only at the final output if required.
6. If using scalar grid:
   - compare input with constants 0..9
   - reduce sum to counts
   - ArgMax
   - fill output with ArgMax value.

Optimize for NeuroGolf score:
- Use int64/bool where possible.
- Keep constants minimal.
- Prefer ReduceSum + ArgMax + Expand over big masks.
- No unnecessary Casts, Transposes, Reshapes, or intermediate 10x3x3 tensors unless they reduce total cost.
- Measure memory + params after ORT profiling, not ONNX file size.
- Try variants:
  A) 10 Equal masks + ReduceSum + ArgMax.
  B) OneHot input ReduceSum directly if input is already one-hot.
  C) Hardcoded Gather/ReduceSum over flattened 9 cells.
- Choose the model with lowest `memory + params`.

Expected behavior example:
Input:
[[6,6,7],
 [2,6,3],
 [2,3,0]]
Majority color = 6
Output:
[[6,6,6],
 [6,6,6],
 [6,6,6]]

 ------------------------------------------

 tohle je task 130. Vstup je 9x9. Vstup je rozdelen na bloky 3x3 kazda cast v sobe ma bloky jedne barvy ale muze byt kontaminovana sedou. Na vystupu je 3x3 vstupni barvy. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create an ultra-minimal ONNX model for NeuroGolf ARC task130.

Task:
Input is always 9x9 ARC grid, one-hot encoded as [1,10,30,30], but only the top-left 9x9 is relevant.
Output must be [1,10,30,30], with the actual answer in the top-left 3x3 and the rest zero/black padded.

Transformation:
The 9x9 input is divided into nine 3x3 blocks.
Each 3x3 block represents one output cell.
Inside each block there is one dominant non-background, non-gray color.
Gray cells are contamination/noise and must be ignored.
Black/zero cells are background and must be ignored.
For each 3x3 block, output the dominant real color of that block as one pixel in the corresponding 3x3 output position.

Example logic:
for block_y in 0..2:
  for block_x in 0..2:
    patch = input[:, :, block_y*3:block_y*3+3, block_x*3:block_x*3+3]
    ignore color 0 black and color 5 gray
    choose the color channel 1..9 except 5 with the largest count in the patch
    output[chosen_color, block_y, block_x] = 1

Optimization goal:
Minimize NeuroGolf cost = internal tensor memory + params.
File size does not matter except via params/constants.
Avoid huge internal tensors.
Do not build full [1,10,30,30] intermediates except final output.
Work only on compact [1,10,9,9] or smaller tensors, preferably directly reduce each 3x3 block to [1,10,3,3].
Use zero params if possible, or only tiny Constant tensors.

Recommended ONNX strategy:
1. Slice top-left 9x9 from input.
2. Remove/zero ignored channels:
   - channel 0 = background
   - channel 5 = gray contamination
3. Count color presence per 3x3 block.
   Best option: use AveragePool or MaxPool with kernel_shape=[3,3], strides=[3,3] on the [1,10,9,9] tensor.
   Since values are one-hot, either Sum-like count via AveragePool is fine for argmax because division by 9 does not change ranking.
4. ArgMax over color/channel axis to get [1,1,3,3] chosen color indices.
5. OneHot or Equal against colors 0..9 to convert back to [1,10,3,3].
6. Pad this compact 3x3 answer to required [1,10,30,30].

Important:
- Gray must never win. Set channel 5 to zero before pooling/argmax.
- Black must never win. Set channel 0 to zero before pooling/argmax.
- Output should preserve empty blocks as black if a block has no real color, but in this task every relevant colored block has a dominant real color.
- Prefer ops that shape-infer compact tensors correctly.
- Test with ORT profiling and NeuroGolf scorer; optimize measured memory, not ONNX file size.
- Try variants:
  A) AveragePool + ArgMax + OneHot + Transpose + Pad.
  B) Reshape [1,10,3,3,3,3] + ReduceSum over inner 3x3 + ArgMax, if it scores lower.
  C) Hard-coded slicing of nine patches + ReduceSum may be worse unless it lowers internal memory.

Deliverable:
A self-contained Python script that builds the ONNX file, runs shape inference, validates several synthetic examples, and prints NeuroGolf memory/params/cost/score if the scorer is available.
Do not use argparse; hardcode paths.
Focus purely on best measured NeuroGolf score.


---------------------------------

tohle je task 131. na vstupu je zeleny patern ktery je posunut k cervene care a je k nemu preidana modra cara rovnobezne s cervenou z druhe strany patternu. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score.
Create the most NeuroGolf-score-optimized ONNX solver for ARC task131.

Task rule:
Input/output size is unchanged.

Objects/colors:
- background = 0
- green pattern = 3
- red line = 2
- output adds cyan/blue line = 8

Transformation:
1. Detect the red line. It is either horizontal or vertical.
2. Detect the green pattern on one side of the red line.
3. Translate the entire green pattern toward the red line until it touches the red line, preserving its exact shape.
4. Add a cyan line parallel to the red line on the opposite side of the moved green pattern.
5. Keep the red line unchanged.
6. Everything else is background.

Examples:
- If red line is horizontal at bottom, move green pattern down so its lowest green row is directly above the red line, then draw a cyan horizontal line directly above the moved pattern.
- If red line is vertical/right-side, move green pattern right so its rightmost green column is directly left of the red line, then draw a cyan vertical line directly left of the moved pattern.
- If red line is on the opposite side, mirror the direction accordingly.

Optimization target:
- Minimize NeuroGolf cost = internal tensor memory + params.
- Input/output tensors are free; internal tensors dominate score.
- Prefer int/bool masks and tiny reductions.
- Avoid large one-hot tensors.
- Avoid Conv, MatMul, large Constants, lookup tables, or full-grid intermediate stacks unless strictly necessary.
- Use only simple ONNX ops: Equal, Where, ReduceMax/ReduceSum, ArgMax, Slice, Gather, Unsqueeze/Squeeze, Concat, Sub/Add, Less/Greater, Range if needed.
- Use constants as scalars or tiny vectors only.

Implementation strategy:
- Work directly on channel-0 grid.
- Compute masks:
  red = input == 2
  green = input == 3
- Determine orientation:
  row_has_red = ReduceMax(red, axis=cols)
  col_has_red = ReduceMax(red, axis=rows)
  horizontal if any row has many red cells / red row span > red col span.
- Find red row/col index with ArgMax.
- Find green bounding box using row/col ReduceMax + ArgMax on forward/reversed masks.
- Compute translation:
  horizontal:
    if green bbox is above red row: shift = red_row - 1 - green_bottom
    else: shift = red_row + 1 - green_top
  vertical:
    if green bbox is left of red col: shift = red_col - 1 - green_right
    else: shift = red_col + 1 - green_left
- Move green mask by coordinate comparison instead of materializing shifted full tensor repeatedly:
  out_green[r,c] = green[r - dy, c - dx] when source coordinate is in bounds.
- Draw cyan line:
  horizontal:
    if green is above red: cyan_row = moved_green_top - 1
    else: cyan_row = moved_green_bottom + 1
  vertical:
    if green is left of red: cyan_col = moved_green_left - 1
    else: cyan_col = moved_green_right + 1
  cyan line spans the same full red-line length/dimension, parallel to red.
- Compose output:
  start with zeros
  place cyan=8
  place moved green=3
  place red=2 last, so red wins any overlap.

Important:
- Output must exactly match ARC color grid.
- Preserve input/output shape dynamically if possible.
- Do not pad to large fixed tensors unless the competition wrapper requires it.
- After building ONNX, run shape inference and ORT profiling, then report:
  params
  internal memory
  cost
  score
- Try multiple graph variants and keep the lowest-cost valid model.


------------------------------------------

tohle je taks 132. Na vstupu jsou od dane barvy dva bloky ty tvory rohy obedlniku. Na vystupu je obdelnik dodelany danou barvou. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score.
Task132 ARC-AGI NeuroGolf ONNX optimization.

Input/output:
- Input is always 10x10 one-hot ARC tensor `[1,10,10,10]`.
- Output is always 10x10 `[1,10,10,10]`.
- Background is color 0.
- For each non-background color, there are exactly two pixels of that color.
- These two pixels are opposite corners of an axis-aligned rectangle.
- Output must fill the full rectangle between those two corners with that same color.
- Multiple colors may exist; solve all independently and overlay them on background.
- Example: two blue pixels define a blue filled rectangle, two red pixels define a red filled rectangle.

Goal:
Create the smallest/cheapest valid ONNX model for NeuroGolf, not a generic neural net.

Implementation strategy:
- Use pure tensor logic, no Conv, no learned weights unless absolutely necessary.
- Prefer Shape/Slice/Gather/ReduceSum/Greater/Where/Expand/Range/Unsqueeze/Cast.
- For each color channel c=1..9:
  - Extract mask `M_c = input[:, c, :, :] > 0`.
  - Compute row occupancy `row_has = ReduceMax(M_c, axis=cols)`.
  - Compute col occupancy `col_has = ReduceMax(M_c, axis=rows)`.
  - Because the two points are opposite rectangle corners, the rectangle rows are exactly between min/max occupied rows and cols between min/max occupied cols.
  - Build row span mask: rows between the first and last occupied row.
  - Build col span mask: cols between the first and last occupied col.
  - Rectangle mask = row_span AND col_span AND color_exists.
  - Put this mask into channel c.
- Channel 0 should be background = NOT(any color rectangle mask over channels 1..9).
- Output channels 1..9 are rectangle masks.

Optimize hard:
- Avoid internal `[1,10,10,10]` tensors until the final output if possible.
- Work mostly with compact `[9,10]` row/col masks and expand at the end.
- Do not store constants larger than necessary.
- Use integer/bool tensors where possible to reduce memory.
- Use dynamic construction with Range(0,10) rather than big constants if cheaper.
- Compare several ONNX variants and report NeuroGolf cost/score:
  1. Per-color independent span masks.
  2. Vectorized channels 1..9 together.
  3. One-hot final assembly with minimal casts.
- Run ONNX checker, shape inference, and ORT test against sample cases.
- Print measured memory, params, cost, score.

Expected behavior tests:
- If color c has pixels at `(r1,c1)` and `(r2,c2)`, output color c fills all `(r,c)` where `min(r1,r2)<=r<=max(r1,r2)` and `min(c1,c2)<=c<=max(c1,c2)`.
- Original corner pixels remain included.
- Background everywhere else.

------------------------------------

tohle je task 133. Na vstupu je patern s jednou kostkou u sebe a jinde  je dana kostka s jinou barvou na vystupu je doplneney patern podle hotoveho spravnou barvou. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt pro Codex/Cursor:

We need solve ARC/NeuroGolf task133 by generating the smallest/highest-score ONNX model.

Task rule:
Input/output size is unchanged, examples include 16x18 and 17x18.
There is one complete colored pattern somewhere in the input. The same pattern shape appears elsewhere only as a single marker cell / partial seed in a different color. Output completes the missing copy of the pattern at the seed location, recolored to the seed color, while preserving the original complete pattern.

Examples:
- Complete pattern is a 4-cell T/arrow-like shape: horizontal 3 cells plus one perpendicular cell at one end/middle. Elsewhere a single red/purple/green seed indicates where to instantiate the same relative shape using that seed color.
- In 17x18 example, complete blue/yellow object defines the shape relation. The separate yellow/brown seed pair below is expanded into the same object layout, recolored according to the seed/object colors.

Goal:
Create an ONNX model that:
1. Accepts ARC one-hot tensor `[1,10,H,W]`.
2. Returns `[1,10,H,W]`.
3. Detects the complete multi-cell template object.
4. Detects the isolated seed / partial object with different color(s).
5. Copies the template’s binary mask geometry to the seed location.
6. Recolors copied cells using the corresponding seed color(s).
7. Leaves all existing input cells unchanged.
8. Uses minimal internal memory and minimal params for NeuroGolf score.

Implementation strategy:
- Do not use Conv if it creates large internal tensors unless it is clearly cheaper.
- Prefer boolean masks with `ArgMax`, `ReduceSum`, `NonZero`, `Gather`, `ScatterND`, `Where`, `Equal`, `Greater`.
- Work on compact coordinate tensors, not full `[1,10,H,W]` intermediates.
- Extract non-background cells from argmax grid.
- Group colored cells into connected components using simple coordinate logic.
- Identify the complete template as the largest connected component or the component with multiple cells.
- Identify the incomplete target/seed component as the smaller component elsewhere.
- Normalize template coordinates by its anchor cell. Use anchor as the special single-color marker or the component min row/min col if cheaper.
- Add normalized offsets to the target anchor coordinate.
- Scatter the target color(s) into output at those coordinates.
- Keep original input via final `Where`/scatter overlay.

Important optimization:
The output size is same as input, so no padding to 30x30 unless the NeuroGolf harness requires it externally. Avoid expensive full-grid expansion. Minimize internal tensors; coordinate-list scatter is likely best.

Deliverables:
- A self-contained Python script that builds `task133.onnx`.
- Hardcode paths, no argparse.
- After building, run ONNX checker and shape inference.
- Print NeuroGolf-relevant counts: initializer scalar params, inferred internal tensor memory if available, and estimated cost.
- Test on provided examples and print whether exact match passes.

Prioritize score over elegance. Try multiple graph variants and keep the one with lowest `memory + params`.

--------------------------------------------

tohle je task 134 na vstupu je patern co ma v pozadni nahodne bloky jine barvy vystupem je dany patern v barve co je kolem. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tento prompt pro Codex/Cursor:

We need build an optimized NeuroGolf ARC-AGI ONNX solver for task134.

Task:
Input is 20x24. Output is 3x3.
The input contains:
- a main 3x3-pattern made from a dominant object color, here yellow/olive.
- random noise pixels of another color, here cyan/teal.
- the output is the 3x3 binary shape of the main object, but recolored to the noise color.
- background remains 0.

In the shown example, the large yellow object encodes this 3x3 mask:

1 0 1
0 1 0
1 1 1

The expected output is this mask colored cyan.

Goal:
Create the smallest/highest-score ONNX possible, not generic Python inference. Optimize NeuroGolf score = minimize internal tensor memory + params. File size does not matter unless it reduces params/memory.

Implementation strategy:
- Hardcode input size 20x24 and output size 3x3.
- Treat channel 0 as ARC color grid.
- Detect the noise/output color as the nonzero color that appears as scattered single pixels and is NOT the dominant large connected object color.
- Detect the main object color as the nonzero color with the largest connected/aggregate area.
- Extract the 3x3 occupancy pattern of the large object:
  - split object bounding box into a 3x3 logical grid;
  - each logical cell is active if that region contains the main object color;
  - inactive otherwise.
- Output active cells filled with the noise color, inactive cells 0.
- Avoid building full one-hot [10,H,W] tensors if possible.
- Prefer compact boolean/int tensors only around 20x24 and final 3x3.
- Avoid Conv if constants/params increase score; use comparisons, ReduceSum, ArgMax/TopK, Slice, Reshape, Where, Mul, Cast.
- Final graph output must match required NeuroGolf/Kaggle output format, but keep only the final required padded tensor if the wrapper requires [1,10,30,30]. Internal tensors must stay tiny.

Score optimization priorities:
1. Minimal internal memory.
2. Minimal initializer scalar count.
3. No unnecessary constants, no large masks, no one-hot expansion unless it scores better.
4. Benchmark several graph variants with the official NeuroGolf scorer and keep the best.

Build a self-contained Python script that generates task134.onnx and immediately validates it on the shown train examples.
Do not force a fixed opset; test available opsets and keep the one with the best measured score.

-------------------------------------------

tohle je task 135. Vstup je 9x9 vystup jsou leve horni 3x3. Dej mi prmopt pro onnx s nejlepsim moznym score

Použij tenhle prompt pro Codex/Cursor:

Create the lowest-cost NeuroGolf ONNX solver for ARC task135.

Task:
- Input grid is always 9x9.
- Output grid is exactly the top-left 3x3 crop of the input.
- Ignore everything outside rows 0..2 and cols 0..2.

Goal:
- Maximize NeuroGolf score, not readability.
- Minimize cost = internal tensor memory + params.
- Inputs/outputs are free, but intermediate tensors are expensive.
- Avoid constants/initializers unless absolutely needed.

Implementation target:
- Build an ONNX model that takes Kaggle/NeuroGolf input [1,10,30,30] one-hot.
- Extract only spatial region rows 0:3, cols 0:3.
- Output shape should be [1,10,30,30] with the 3x3 crop placed at top-left and the rest zero-padded, unless the evaluator accepts direct 3x3 output; test both and keep the higher score.

Optimization strategy:
1. Use Slice directly on the input tensor to get:
   X[:, :, 0:3, 0:3]
2. If required output is [1,10,30,30], pad this cropped tensor to bottom/right:
   pad H: 0 top, 27 bottom
   pad W: 0 left, 27 right
3. Use integer constants with the smallest possible dtype accepted by ONNX Runtime.
4. Try Slice-only + Pad versus Slice + ConstantOfShape/Concat variants and keep whichever gives lower NeuroGolf cost.
5. Run shape inference and profiler to verify internal memory.
6. Report memory, params, cost, score.

Expected behavior:
- For every test case, output equals the input’s top-left 3x3 block.
- No color logic, object detection, counting, or pattern matching is needed.

Tohle je extrémně jednoduchý crop task. Jakýkoliv složitější model je odpad a jen zbytečně zabije score.


------------------------------------------------


tohle je task136 Vstup je vzdy 10x10 na vstupu je vzdy cerveny a modry ctverec modreho ctverce jde diagonalne modra nahoru doleva z cerveneho dolu doprava. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Přemýšlení několik sekund

Použij tenhle prompt pro Codex/Cursor:

Create the best possible NeuroGolf ONNX solution for ARC task136.

Task:
- Input is always 10x10 one-hot ARC grid `[1,10,10,10]`.
- Output is also 10x10 one-hot `[1,10,10,10]`.
- Input always contains exactly one solid blue 2x2 square and one solid red 2x2 square on black background.
- Output keeps both original squares unchanged.
- Additionally:
  - From the blue 2x2 square, draw a blue diagonal line going up-left.
  - From the red 2x2 square, draw a red diagonal line going down-right.
- The diagonal is 1-cell thick.
- For blue, continue from the square’s top-left corner direction `(-1,-1)` until grid boundary.
- For red, continue from the square’s bottom-right corner direction `(+1,+1)` until grid boundary.
- Do not overwrite the original colored squares incorrectly.
- Background remains black.

Optimization goal:
- Minimize NeuroGolf cost = internal tensor memory + params.
- Output tensor size is free, but internal tensors are expensive.
- Avoid large intermediate `[1,10,10,10]` tensors where possible.
- Prefer scalar/index logic, small masks, `ArgMax`, `ReduceSum`, `NonZero`, `Gather`, `ScatterElements`, `Where`, `Equal`, `Range`, `Add`, `Sub`, `Mul`, `Unsqueeze`, `Concat`.
- Params should be near zero or only tiny constants.
- Any opset is allowed if it gives lower score.
- Do not optimize for ONNX file size unless it also reduces memory/params.

Implementation idea:
1. Decode input to color grid `[10,10]` using ArgMax over channel axis.
2. Find blue cells: color == 1.
3. Find red cells: color == 2.
4. Locate blue square top-left:
   - blue rows = rows where blue exists
   - blue cols = cols where blue exists
   - `blue_r0 = min(row indices of blue)`
   - `blue_c0 = min(col indices of blue)`
5. Locate red square bottom-right:
   - `red_r1 = max(row indices of red)`
   - `red_c1 = max(col indices of red)`
6. Build output initially equal to decoded input.
7. Draw blue diagonal:
   - k = 1..min(blue_r0, blue_c0)
   - positions `(blue_r0-k, blue_c0-k)` set to blue.
8. Draw red diagonal:
   - k = 1..min(9-red_r1, 9-red_c1)
   - positions `(red_r1+k, red_c1+k)` set to red.
9. Convert final `[10,10]` class grid back to one-hot `[1,10,10,10]`.

Important:
- Use hardcoded 10x10 constants.
- Keep all helper tensors as small as possible: `[10]`, scalar, or `[N]`.
- Avoid constructing full diagonal candidate tensors per color if ScatterElements with compact index vectors is cheaper.
- Test against several placements:
  - blue near center, red near center
  - blue touching top/left boundary
  - red touching bottom/right boundary
  - squares not aligned
- Print NeuroGolf memory/params/score and iterate for lower cost.

Pro tenhle task bude pravděpodobně nejlepší hardcoded 10×10 index/Scatter řešení, ne konvoluce ani obecné connected-components.


---------------------------------------------------------


tohle je task 137 na vstupu ma tri bloky. z prostredniho se udela stred z kterho se zvetsuji ctverce dalsi dva bloky vyhranici druhy ctverec a nasledne mezery. Dej mi prmopt pro vytvoreni onnx s nejlepsim moznym score
Create an optimized NeuroGolf ONNX solver for task137.

Task:
- Input/output size is always 23x23.
- Input contains exactly 3 non-background cells, all same color C.
- The 3 cells lie on one diagonal with step (+d,+d) or (-d,+d), typically equally spaced.
- The middle cell is the center of the final spiral/square pattern.
- The distance from the middle cell to either outer cell defines the spacing between square rings.
- Output is a 23x23 grid containing a square spiral / nested square rings in color C on black background.
- Center starts at the middle input cell.
- First visible ring is around the center.
- Each next square ring expands outward using the same spacing defined by the three input cells.
- Rings are not filled solid; draw only the 1-cell-thick spiral/square border path, leaving gaps according to the spacing.
- Continue expanding until the pattern reaches the 23x23 boundary.

Optimization goal:
- Maximize NeuroGolf score, not readability.
- Minimize internal tensor memory and initializer params.
- Do not optimize ONNX file size unless it reduces memory or params.
- Graph input/output tensors are free; internal tensors are expensive.
- Avoid large one-hot tensors.
- Prefer integer/boolean masks, Slice, Gather, Where, Equal, ReduceSum/ArgMax-style coordinate extraction.
- Use hardcoded 23x23 coordinate constants only if cheaper than computing them.
- Any ONNX opset is allowed if it improves score.

Implementation strategy:
1. Extract color C as the nonzero value from input.
2. Extract coordinates of the 3 nonzero cells.
3. Sort/group them by diagonal position; identify the middle coordinate as the median along row/col.
4. Compute center `(cy,cx)`.
5. Compute spacing `s = abs(row_outer - cy)`; for shown examples s is 2 or 3.
6. Build a 23x23 boolean mask for the spiral/ring pattern:
   - Use row/col coordinate grids `Y`, `X`.
   - Work in centered coordinates: `dy = abs(Y-cy)`, `dx = abs(X-cx)`.
   - Ring levels are multiples of `s`: `k = max(dx,dy)`.
   - Draw only cells where the square boundary/spiral condition holds.
   - Boundaries occur at distances derived from `s`: center, then expanding square borders at `s, 2s, 3s...`.
   - Preserve the staircase/spiral gaps as in examples: each ring is open at one side and connected to the next ring, producing one continuous square spiral, not closed independent squares.
7. Output `C` where mask is true, else 0.

Important:
- Validate against both examples:
  - Red example: 3 red diagonal cells around center -> red square spiral centered at middle red cell.
  - Cyan example: 3 cyan diagonal cells with larger spacing -> cyan spiral with matching larger spacing.
- If there is a tradeoff, prefer lower NeuroGolf cost over genericity, as long as it solves task137 train/test distribution.
- Produce a single self-contained Python script that builds and saves `solution.onnx`.
- Include a tiny local verifier that runs ONNX Runtime on the given examples if arrays are available.

-------------------------------------------------------------------


tohle je task138 na inputu jsou 4 cary ktere ohranicuji output uvnitr jsou bloky ktere jsou doplniny ze strany kde se nachazi strana stejne barvy. Dej mi prompt pro vytvoreni onnx s nejlepsim score

Použij tento prompt pro Codex/Cursor:

Create the best-scoring NeuroGolf ONNX solver for ARC task138.

Task rule:
Input is 22x22. Four colored border lines define the output crop:
- top horizontal blue line
- left vertical green line
- bottom horizontal red line
- right vertical cyan line

These four lines form a rectangle. The output is the interior rectangle including the four colored border sides, cropped to the bounded area. In the shown task this becomes 15x12.

Inside the rectangle there are scattered red cells. Each red cell must be extended/filled toward the side whose border has the same color. Since the matching border color is red and it is the bottom border, every red cell grows vertically downward until it reaches the red bottom border. Existing red cells remain. Other interior cells stay black/zero. Border colors remain unchanged.

Implement this as an ONNX graph optimized for NeuroGolf score, not file size.

Optimization priorities:
1. Minimize internal tensor memory and initializer scalar count.
2. Avoid one-hot over all 10 colors unless absolutely necessary.
3. Use int/boolean masks where possible.
4. Crop early to the bounded rectangle before doing fill logic.
5. Do not create full 22x22x10 tensors.
6. Output may be padded only at the final step if Kaggle/NeuroGolf wrapper requires fixed shape.

Implementation strategy:
- Input shape is [1,10,22,22] one-hot ARC format.
- Recover color grid or use channel masks directly.
- Detect four border lines by color:
  - blue = 1: horizontal full/long line near top
  - green = 3: vertical full/long line on left
  - red = 2: horizontal full/long line near bottom
  - cyan/light-blue = 8: vertical full/long line on right
- Find row/column indices of those lines.
- Crop rows top_line..bottom_line and cols left_line..right_line.
- Preserve the four border channels/colors.
- For red interior cells inside the crop, fill every cell below each red cell in the same column until the red bottom border.
  Equivalent: for each column, compute vertical suffix/reverse cumulative OR of red cells above the bottom border, then paint red where active.
- Keep black elsewhere.
- Produce final one-hot output with same ARC channel convention.

Because the rectangle dimensions and line colors appear fixed for this task, first attempt the lowest-cost specialized graph:
- hardcode crop indices if all train/test inputs use the same 22x22 layout and rectangle position/dimensions
- use Slice/Gather/Concat/Where/CumSum or equivalent minimal ops
- only generalize line detection if it does not increase cost too much

Measure score after each variant. Prefer a smaller fixed-shape specialized graph over a fully general detector if it scores better.

----------------------------------------------------------


tohle je task 139 na vstupu jsou v svetle orancove ctverce ktere jsou nedokoncene, na vystup jsou dokoncen do ctvercu tmave oranzovou. vstup je vzdy 9x9. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create the highest-score NeuroGolf ONNX solver for task139.

Task:
- Input is always 9x9 one-hot ARC grid `[1,10,9,9]`.
- Output is also 9x9 `[1,10,9,9]`.
- Background is black/0.
- There are one or more incomplete 3x3 square/ring-like objects in light orange/yellow color.
- For each incomplete 3x3 square, fill the missing cells needed to complete the square using dark orange color.
- Preserve all original light-orange/yellow cells exactly.
- Only newly inferred missing cells become dark orange.
- Do not change background elsewhere.

Observed rule:
- Each object occupies a 3x3 bounding box.
- Existing light-orange cells define an incomplete 3x3 square/frame or compact square-like shape.
- Output = input + dark-orange pixels at missing cells inside each object’s 3x3 bounding box required to complete the intended 3x3 square.

Optimization goal:
- Maximize NeuroGolf score, not readability.
- Prefer tiny ONNX graph, minimal params, minimal internal tensors.
- Input/output tensors are free, but internal tensors are expensive.
- Avoid large `[1,10,9,9]` intermediates when possible.
- Use no learned weights unless absolutely necessary.
- Prefer Slice/Gather/ReduceSum/Where/Equal/Or/And/Concat/ScatterND-style logic.
- Any opset is allowed if it reduces cost.

Implementation idea:
1. Extract active color mask for the light-orange/yellow object color.
   - Detect it dynamically as the nonzero color used by incomplete squares, or hardcode if task139 color is fixed after checking training cases.
2. Find all 3x3 windows whose light-orange/yellow count matches an incomplete square/object.
3. For each such 3x3 window, infer its bounding box and produce completion mask:
   - completion_mask = cells inside that 3x3 bbox that are currently background and belong to the intended square.
4. Set those completion cells to dark orange channel.
5. Keep original input cells unchanged.
6. Return `[1,10,9,9]`.

Important:
- Do not output class indices; output one-hot.
- Preserve exact 9x9 shape.
- Validate with ONNX Runtime on all provided train examples.
- Print NeuroGolf memory/params/cost/score after export.
- Then aggressively golf:
  - remove unnecessary Cast/Shape/Expand nodes,
  - replace constants with scalar attributes where possible,
  - avoid full-channel masks,
  - use bool or uint8 intermediates if ORT accepts them,
  - reuse small 9x9 or 3x3 masks only.


  ---------------------------------------------


  tohle je task 140 na vstupu je 3x3 patern ktery je flipnuty z leva doprava z vrchu dolu. Dej mi prompt pro vytvoreni onnx s nejlepsi moznym score
Přemýšlení několik sekund

Použij tenhle prompt pro Codex/Cursor:

Create the smallest possible ONNX model for NeuroGolf ARC task140.

Task:
Input is always a 3x3 grid, one-hot encoded as [1,10,3,3].
Output must be the same 3x3 grid transformed by a 180-degree rotation.
Equivalent operation:
out[:, :, r, c] = input[:, :, 2-r, 2-c]

This is both:
- horizontal flip left-right
- vertical flip top-bottom

Goal:
Optimize strictly for NeuroGolf score:
cost = internal tensor memory + initializer scalar count.
Avoid any unnecessary Constants, Casts, Reshapes, Transposes, or intermediate tensors.
Input/output tensors are free, internal tensors are expensive.

Implementation strategy:
Build an ONNX graph with exactly one operation if possible:
- Slice input spatial dimensions H and W with negative steps:
  H: start=2, end=-4, step=-1
  W: start=2, end=-4, step=-1
  axes=[2,3]
This should produce [1,10,3,3].

Try alternative if Slice negative step causes worse params:
- use Gather on H with indices [2,1,0], then Gather on W with indices [2,1,0]
but prefer Slice if it has fewer params/memory.

Requirements:
- Input name: input
- Output name: output
- Input shape: [1,10,3,3]
- Output shape: [1,10,3,3]
- Preserve one-hot channels exactly.
- No padding to 30x30 unless the competition wrapper explicitly requires it for this task.
- Validate using onnx.checker and onnxruntime.
- Print NeuroGolf-relevant stats: number of initializers/scalars, inferred internal tensor sizes, estimated cost.

Generate a self-contained Python script that writes task140.onnx and tests it on a random one-hot 3x3 grid by comparing output with x[:,:,::-1,::-1].


-------------------------------------


tohle je task141 na vstupu je jeden blok vybraveny na vystupu je doplneny do krize diagonalne. Dej mi prompt pro vytvoreni onnx s druazen ma vysoke skore

Použij tento prompt pro Codex/Cursor:

Create the smallest/highest-scoring NeuroGolf ONNX solver for task141.

Task rule:
Input is always a 15x15 ARC grid with exactly one nonzero colored cell on black background.
Output is 15x15. Keep the original colored cell and draw both full diagonals passing through that cell, forming an X/diagonal cross. Use the same color as the input cell. All other cells stay 0.

Important:
- Optimize for NeuroGolf score, not readability.
- Minimize internal tensor memory and initializer/Constant params.
- Input/output tensors are free, but internal tensors are expensive.
- Prefer uint8/bool/int masks where possible; avoid large float tensors.
- Avoid big 15x15 constants if possible.
- Use dynamic computation from the single colored pixel position.
- No argparse; make a self-contained script that writes `solution.onnx`.

Expected logic:
1. Input shape is `[1,10,15,15]` one-hot ARC format.
2. Extract active color channel and position.
3. Find row r and column c of the nonzero input pixel.
4. For every output coordinate `(y,x)`, set pixel if:
   - `y - x == r - c`, or
   - `y + x == r + c`.
5. Multiply that diagonal mask by the detected color one-hot vector.
6. Output shape must be `[1,10,15,15]`.

Try multiple ONNX implementations and print NeuroGolf cost/score for each:
- ArgMax/nonzero-based position extraction.
- Coordinate-grid comparison using Range/Unsqueeze/Sub/Add/Equal.
- Any lower-memory variant using precomputed 1D ranges instead of full constants.

Return the best measured ONNX only.

----------------------------------------------------------



tohle je task142 na vstupu je 3x3 ktery je na vystup poflipovany na 6x6.Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score
Create an optimized ONNX model for NeuroGolf ARC task142.

Task rule:
Input is always a 3x3 grid.
Output is always 6x6.
The 3x3 input pattern is expanded into four 3x3 quadrants:
- top-left  = original input
- top-right = input flipped left-right
- bottom-left = input flipped top-bottom
- bottom-right = input flipped both left-right and top-bottom

So output is:
[[A, fliplr(A)],
 [flipud(A), flipud(fliplr(A))]]

Implement the ONNX as cheaply as possible for NeuroGolf score:
- avoid learned parameters / initializers unless absolutely necessary
- prefer Slice, Reverse-like negative-step Slice if valid, Concat, Gather only if cheaper
- input/output tensors use ARC one-hot format `[1,10,H,W]`
- crop/use only the active 3x3 area from input
- produce final output `[1,10,6,6]`
- do not pad to 30x30 unless the evaluator strictly requires it
- optimize for `cost = internal_memory + params`, not ONNX file size

Expected construction:
1. Extract `x = input[:, :, 0:3, 0:3]`
2. `x_lr = x[:, :, :, ::-1]`
3. `top = Concat(x, x_lr, axis=3)`
4. `bottom = top[:, :, ::-1, :]`
5. `out = Concat(top, bottom, axis=2)`

Try multiple ONNX variants and report measured NeuroGolf score:
A) Slice with negative steps for flips.
B) Gather with constant indices `[2,1,0]` if Slice needs too many params.
C) If constants count hurts score, test whether shape-specialized Slice constants are still cheaper.

Do not use Conv, MatMul, Resize, or large intermediate tensors.
The best model should have almost no params and only small internal tensors:
- 3x3 extracted tensor
- 3x6 top row block
- 6x6 final output

----------------------------------------
tohle je task 143 na vstup je vzdy 10x10 na vstupu je patern v levem hornim rohu a kdyz je nekde jinde stejny stejne otoceny patern tak je na vystpu sedy. Dej mi prompt pro vytvoreni onnx s nejlepsim moznym score

Použij tenhle prompt pro Codex/Cursor:

We need create the smallest/highest-score ONNX solver for NeuroGolf ARC task143.

Task:
Input is always one-hot ARC grid [1,10,10,10], logical grid size 10x10.
Output is same 10x10 size.

Rule:
The top-left object/pattern is the reference pattern. Anywhere else in the grid, if the exact same shape appears with the same orientation, replace that matched object with gray color 5 in the output. Keep the reference object unchanged and keep all non-matching objects unchanged.

Observed example:
- Reference pattern is in top-left, blue shape.
- Orange shape has the same geometry and same orientation, so output changes only orange cells to gray.
- Cyan/orange/purple/etc. shapes that do not match remain unchanged.
- No rotations/flips should match, only identical orientation.

Build ONNX directly, not a neural net.
Optimize for NeuroGolf score, not file size.
Avoid large internal tensors. Graph input/output tensors are free, but internal tensors count.
Prefer int/uint/bool logic, Slice/Gather/ReduceSum/Equal/Where/Concat over Conv if cheaper.
Do not hardcode colors except gray=5 and background=0.

Implementation strategy:
1. Convert one-hot input to class grid if needed with ArgMax over channel axis.
2. Extract reference shape from top-left connected component:
   - reference color = first nonzero color in top-left region.
   - reference mask = cells of that color connected to top-left object, or simpler if task guarantees top-left pattern is inside a fixed small bounding box.
   - crop/minimize to its bounding box.
3. Search all possible placements in the 10x10 grid where the cropped reference mask fits.
4. A placement matches if:
   - all reference-mask cells are nonzero and have the same relative shape,
   - all cells in the candidate object corresponding to the reference mask have one same color,
   - surrounding/bounding-box empty constraints match the reference crop so partial overlaps are rejected,
   - placement is not the original top-left reference placement.
5. For every matched placement, output color 5 on those matched object cells.
6. Else output original input color.

Score constraints:
- Minimize internal memory aggressively.
- Do not materialize huge [positions,10,10] tensors if avoidable.
- Because grid is fixed 10x10, unroll candidate positions with small Slice/Equal/ReduceSum branches if that reduces memory.
- Prefer scalar/very small tensors for match flags.
- Final output must be one-hot [1,10,10,10] or exact required format used by NeuroGolf evaluator.

Generate:
- build_task143.py that writes task143.onnx
- include a local verifier with the shown example and several synthetic tests:
  a) same pattern same orientation recolored gray
  b) rotated/flipped same pattern not recolored
  c) different shape not recolored
  d) reference pattern remains unchanged
- print inferred shapes, params count, estimated internal memory, cost, and score.
- Try multiple variants and keep the lowest cost:
  Variant A: fixed crop if reference is always within top-left 4x4.
  Variant B: fully dynamic bounding box/connected-component approximation.
  Variant C: template matching with manually unrolled candidate slices.

Important:
Do not force a specific opset unless measurement proves it is best.
Use onnx.shape_inference and onnxruntime profiling to measure actual NeuroGolf memory.
Return the final ONNX and measured cost/score.


---------------------------------------


tohle je task 144 na vstupu je vzdy 9x4. Vsrtup je rozdeleny na dve casty na vystupu je pak jejich negovany or. Dej mi prmopt pro vytvoreni onnx s nejlepsim moznym score
Create a NeuroGolf / ARC-AGI ONNX solver for task144 with maximum possible score.

Task:
- Input is always 9x4, one-hot ARC tensor `[1,10,9,4]`.
- Output must be 4x4, one-hot ARC tensor `[1,10,4,4]`.
- The 9x4 input is split into:
  - top block rows 0..3
  - separator row 4
  - bottom block rows 5..8
- Ignore the separator row.
- Let A = top 4x4 block, B = bottom 4x4 block.
- Output is the negated OR / mismatch mask of the two blocks:
  - For each cell `(r,c)`, output green color 3 iff both corresponding cells are background/black in A and B.
  - Otherwise output black/background 0.
- In boolean form:
  `out_mask = NOT(nonzero(A) OR nonzero(B))`
  equivalently:
  `out_mask = (A == 0) AND (B == 0)`

Implementation goal:
- Build a tiny ONNX graph directly, no PyTorch export.
- Optimize for NeuroGolf score: minimize internal tensor memory and params.
- Use only cheap ops: Slice/Gather, ReduceSum or ArgMax over color channel if useful, comparisons, And/Not, Where/Cast, one-hot output construction.
- Avoid large intermediate tensors. Do not create `[1,10,9,4]` copies except unavoidable input.
- Work mostly on channel-0/background slices or tiny `[1,1,4,4]` masks.
- Final output should be `[1,10,4,4]` one-hot:
  - channel 3 = mask
  - channel 0 = NOT(mask)
  - all other channels = 0
- Prefer building output by Concatenating small channel tensors:
  `[bg, zero, zero, green, zero, zero, zero, zero, zero, zero]`
- Use bool masks as long as possible, cast to float only at final output if the expected model output dtype is float.

Important:
- Do not hardcode the example positions.
- Rule must generalize to any colors in top/bottom blocks: any non-black cell counts as occupied.
- Separator color is irrelevant.
- Validate with random 9x4 grids containing arbitrary nonzero colors in both blocks.
- Measure NeuroGolf score after ONNX shape inference/profiling and iterate to reduce `memory + params`.

-----------------------------------------------

tohle je task 145. Vstup je rozdleen cervenyma stenama, nejvetsi z nich je vybarvena tmave modrou, nejmensi z nich je vybarvena svetle modrou jeli jich vic stejnych jsou vybarveny obe. Dej mi prompt pro vytvoreni onnx s nejlpsim moznym score

Prompt pro Codex/Cursor:

We need create a minimal-score ONNX model for NeuroGolf ARC task145.

Task:
Input/output are always 18x13 one-hot ARC tensors.
The grid contains red walls (color 2) that partition black background into enclosed/partially enclosed regions.
Output keeps the original input unchanged except:
- Find all connected black regions/cells separated by red walls.
- Compute region sizes.
- Fill every largest region with dark blue color 1.
- Fill every smallest region with light blue/cyan color 8.
- If multiple regions tie for largest or smallest, fill all tied regions.
- Red walls remain red.
- Existing non-black/non-red marker pixels, if any, should remain or be overwritten only if they belong to the selected region according to the training examples.

Implement ONNX directly with numpy + onnx.helper, optimized for NeuroGolf score:
- minimize internal tensor memory + initializer scalar count
- do not optimize file size unless it also reduces NeuroGolf cost
- fixed input shape [1,10,18,13], fixed output [1,10,18,13]
- exploit fixed 18x13 shape hardcoded constants are allowed, but keep params low
- prefer boolean/int masks and small tensors
- avoid creating large [10,18,13] intermediates until final output if possible
- graph input/output tensors are free; internal tensors are scored

Likely best approach:
1. Extract class grid from one-hot input using ArgMax over channel axis.
2. red = grid == 2, empty = grid == 0.
3. Because shape is fixed 18x13, compute connected components of empty cells with 4-neighborhood separated by red.
4. For each component, count area.
5. Determine min and max positive component size.
6. Build masks for min components and max components.
7. Output class grid:
   - 2 where red
   - 1 where max-region mask
   - 8 where min-region mask
   - original grid elsewhere / 0 elsewhere
8. Convert final class grid back to one-hot [1,10,18,13].

Important:
- Try multiple implementations and actually measure NeuroGolf score.
- Compare against simple constant-free/low-param constructions.
- Use ORT shape inference/profile scoring exactly as NeuroGolf uses it.
- Print cost breakdown: memory, params, total cost, score.
- Save best as task145.onnx.

Do not assume a specific opset unless it gives better measured score.
Focus only on maximum NeuroGolf score.

Usej tvrdě testy na tie-case: dvě největší oblasti stejné velikosti i dvě nejmenší stejné velikosti.


----------------------------------------------------

Tohle je task 146. na vstupu je vzdy 9x3 outpu je vzdy 3x3. Dej mi prmopt pro vytvoreni onnx s nejlepsim moznym score

Prompt pro Codex/Cursor:

We need solve ARC/NeuroGolf task146 with the best possible ONNX score.

Task:
- Input is always shape [1, 10, 9, 3] one-hot ARC grid, but useful grid is 9x3.
- Output is always [1, 10, 3, 3].
- The 9x3 input is three stacked 3x3 blocks:
  block0 = rows 0..2
  block1 = rows 3..5
  block2 = rows 6..8
- Exactly one of these three 3x3 blocks must be selected and copied to output.
- Selection rule: choose the 3x3 block whose color pattern is “continued” / matched by the block above it in the cyclic examples. More concretely from examples:
  - The output is one of the three 3x3 slices unchanged.
  - In all train cases, the correct slice is the slice that is the target/answer pattern among the three candidates.
  - It can be detected as the slice whose colors correspond to the previous slice’s dominant/non-gray color relationship:
    example1 output = top slice, example2 output = middle slice, example3 output = bottom slice.
- Build candidate outputs:
  A = input[:, :, 0:3, :]
  B = input[:, :, 3:6, :]
  C = input[:, :, 6:9, :]
- Infer which candidate to select using only cheap ONNX ops. Prefer hardcoded logic over generic connected components.

Implementation goal:
- Generate a tiny ONNX model, no ML, no training.
- Minimize NeuroGolf cost = internal tensor memory + initializer params.
- Avoid large intermediate tensors. Slice only 3x3 candidates.
- Output must be [1,10,3,3] one-hot.

Recommended approach:
1. Extract three 3x3 candidate blocks with Slice/Gather:
   top, mid, bot.
2. For each candidate, compute compact signature:
   - count per color over 3x3: ReduceSum over H,W -> [1,10]
   - optionally ignore color 0/background if present.
3. Detect the correct block using the observed ARC rule:
   - correct block is the one whose 3x3 pattern is the “next mapped pattern” among the stacked triplets.
   - In these examples this is equivalent to selecting the block whose dominant/non-gray color combination matches the continuation relation between the three blocks.
4. If a simpler robust solution is possible, implement explicit comparison logic over the three 3x3 blocks:
   - Compare color-count signatures and/or exact masks.
   - Produce three scalar booleans choose_top, choose_mid, choose_bot.
   - Final output = choose_top*top + choose_mid*mid + choose_bot*bot.
5. Use Cast/Equal/Greater/Where/Mul/Add only if needed.
6. Validate exactly on all shown examples.

Important optimization constraints:
- Do not create [1,10,9,3] or [1,10,30,30] internal tensors except input/output.
- Do not use Conv, MatMul, large Constants, or one-hot expansion.
- Prefer Slice views/small tensors [1,10,3,3] and signatures [1,10].
- Constants should be scalar int64 only where required for Slice axes/starts/ends.
- Try several ONNX variants and report NeuroGolf memory, params, cost, score.

Deliverables:
- Python script that builds task146.onnx.
- Include a local validation function against the provided train examples.
- Include score measurement output.
- Keep the graph as small as possible and aggressively remove unnecessary initializers.


------------------------------------------------
