# SAM3 to YOLO Auto Label Pipeline Guide

This document explains how to prepare data, install dependencies, run the pipeline, inspect outputs, and troubleshoot common issues.

The pipeline script is `auto_label_pipeline.py`. It uses SAM3 text prompts to generate segmentation masks, converts masks into YOLO segmentation polygons, and writes a ready-to-use dataset under `runs/`.

Additional prompt references:

- Chinese prompt guide: `prompt-guide-zh.md`
- reusable prompt templates: `prompt-examples.md`

## What the Pipeline Does

Given a resource directory such as `src/`, the pipeline will:

1. read class definitions from `prompts.yaml`
2. process images in `images/` directly
3. extract frames from videos in `videos/` with `ffmpeg`
4. run SAM3 prompt-based inference on each image or frame
5. convert masks into YOLO segmentation label files
6. write dataset images, labels, a dataset config, and a manifest

## Directory Layout

Expected input layout:

```text
src/
  images/
  videos/
  prompts.yaml
```

Current repository example:

```text
sam3/
  auto_label_pipeline.py
  checkpoints/
    sam3.pt
  src/
    images/
    videos/
    prompts.yaml
```

## Requirements

The pipeline needs:

- Python 3.8+
- PyTorch
- OpenCV
- Pillow
- PyYAML
- `pycocotools`
- `iopath`
- `ffmpeg`
- a SAM3 checkpoint file, typically `checkpoints/sam3.pt`

GPU is recommended. CPU mode is supported but will be much slower.

## Install

From the `sam3` repository root:

```bash
pip install -e .
pip install opencv-python pillow pyyaml pycocotools iopath requests pytest
```

If you plan to use the local tests, `pytest` is recommended as shown above.

## Checkpoint Setup

The pipeline resolves checkpoints in this order:

1. `--checkpoint /path/to/file.pt`
2. local cache at `checkpoints/sam3.pt`
3. download from `--checkpoint-url`

Default remote checkpoint URL:

```text
https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt
```

If you already have the weight locally, put it at `checkpoints/sam3.pt` or pass `--checkpoint` explicitly.

## Input Preparation

### Images

Put image files into `src/images/`.

Supported formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`
- `.webp`

### Videos

Put video files into `src/videos/`.

Supported formats:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.webm`
- `.mpeg`
- `.mpg`

Videos are extracted into frames first, then each frame is labeled as an image.

### Prompt File

Create `src/prompts.yaml` like this:

```yaml
classes:
  - name: person
    prompt: person
  - name: bed
    prompt: bed
  - name: pillow
    prompt: pillow
  - name: lamp
    prompt: lamp
```

Rules:

- `name` becomes the class name in `dataset.yaml`
- `prompt` is the text query sent to the SAM3 processor
- order matters; the first class is `0`, the second is `1`, and so on
- each `classes` item can also be a plain string, which means `name == prompt`

The bundled sample video in this repository is an indoor bedroom scene, so prompts like `bed`, `pillow`, and `lamp` are better starter defaults than traffic or vehicle classes.

## How to Configure Prompts

Prompt quality has a direct effect on label quality. In practice, the prompt is your task definition for each class.

### Core Rule

Use English prompts by default.

Reason:

- English prompts are usually more stable for this type of vision-language model
- category boundaries are often clearer in English
- scene constraints and object descriptions combine more naturally in English

Recommended pattern:

- `name`: the label name you want in the output dataset
- `prompt`: the English phrase the model actually uses for detection

Example:

```yaml
classes:
  - name: person
    prompt: person in a bedroom
  - name: bed
    prompt: bed in an indoor bedroom scene
  - name: pillow
    prompt: pillow on the bed
```

If you want output class names in Chinese, keep `prompt` in English:

```yaml
classes:
  - name: 人
    prompt: person in a bedroom
  - name: 床
    prompt: bed in an indoor bedroom scene
  - name: 枕头
    prompt: pillow on the bed
```

In this setup:

- `name` controls the class name written to `dataset.yaml`
- `prompt` controls what the model tries to find

### Prompt Writing Formula

A good prompt usually contains one or more of these parts:

```text
target object + scene/background + position/detail + appearance constraint
```

Examples:

- `person`
- `person in a bedroom`
- `pillow on the bed`
- `table lamp beside the bed`
- `white car on a street`
- `forklift in a warehouse`

### Good Prompt Characteristics

- specific, not vague
- visually concrete
- close to what actually appears in the image or video
- distinct from other classes in the same prompt set

Better examples:

- `pillow on the bed`
- `bedside table`
- `worker wearing a safety helmet`
- `red forklift in a warehouse`

Weaker examples:

- `object`
- `thing`
- `equipment`
- `indoor item`

### How to Add Background or Goal Information

If you want to tell the model about the background, task, or scene, put that information directly into each class prompt.

Examples:

```yaml
classes:
  - name: person
    prompt: person in an indoor bedroom scene
  - name: bed
    prompt: bed in an indoor bedroom scene
  - name: lamp
    prompt: lamp in a bedroom near the bed
```

For a warehouse example:

```yaml
classes:
  - name: forklift
    prompt: forklift in a warehouse
  - name: pallet
    prompt: wooden pallet on the warehouse floor
  - name: worker
    prompt: warehouse worker wearing a safety vest
```

For a street scene example:

```yaml
classes:
  - name: car
    prompt: car on an urban street
  - name: traffic_light
    prompt: traffic light at an intersection
  - name: pedestrian
    prompt: pedestrian crossing the road
```

### How to Choose the Right Granularity

Use prompts that match the detail level you really need.

If you only care about broad classes:

```yaml
classes:
  - name: person
    prompt: person
  - name: car
    prompt: car
```

If you need more specific categories:

```yaml
classes:
  - name: delivery_van
    prompt: delivery van on a street
  - name: sedan
    prompt: sedan car on a street
```

Be careful not to create overlapping classes unless you really want them. For example, `car` and `sedan` in the same run may compete with each other.

### Prompt Design Tips

- Start with 2 to 5 classes first.
- Use nouns the model is likely to know well.
- Add scene words only when they help disambiguate.
- Add appearance words only when they are visually reliable.
- Prefer `pillow on the bed` over `soft bedroom sleeping object`.
- Prefer `worker wearing a yellow helmet` over `worker with safety gear`.

### Chinese vs English Recommendation

Recommended:

```yaml
classes:
  - name: person
    prompt: person in a bedroom
```

Also acceptable:

```yaml
classes:
  - name: 人
    prompt: person in a bedroom
```

Less recommended:

```yaml
classes:
  - name: 人
    prompt: 卧室里的人
```

If you want the most stable behavior, keep `prompt` in English.

### Recommended Workflow for Prompt Tuning

1. write a small prompt set in `src/prompts.yaml`
2. run the pipeline on a few images or a short video clip
3. inspect several generated labels
4. remove vague prompts and rewrite weak ones
5. rerun until the outputs are stable enough

### Common Prompt Problems

`Too broad`

- `object`
- `furniture`
- `machine`

These often produce noisy masks or unstable categories.

`Too abstract`

- `danger`
- `work area`
- `resting zone`

These are not concrete visual objects, so segmentation quality is usually poor.

`Too overlapping`

- `car`
- `vehicle`
- `sedan`

These may compete with each other and create duplicate or inconsistent labels.

### Prompt Templates You Can Reuse

Indoor room scene:

```yaml
classes:
  - name: person
    prompt: person in an indoor room
  - name: bed
    prompt: bed in a bedroom
  - name: pillow
    prompt: pillow on the bed
  - name: lamp
    prompt: lamp beside the bed
```

Factory or warehouse scene:

```yaml
classes:
  - name: worker
    prompt: factory worker
  - name: helmet
    prompt: safety helmet
  - name: forklift
    prompt: forklift in a warehouse
  - name: pallet
    prompt: pallet on the floor
```

Road traffic scene:

```yaml
classes:
  - name: car
    prompt: car on a road
  - name: bus
    prompt: bus on a city street
  - name: bicycle
    prompt: bicycle on the road
  - name: traffic_light
    prompt: traffic light at an intersection
```

### Quick Checklist Before Running

Before you launch the pipeline, check that:

- every class has a clear visual object
- every `prompt` is written in English
- prompts are specific enough to separate classes
- prompts match the actual scene in your data
- you are not mixing heavily overlapping categories unless intentional

Plain string example:

```yaml
classes:
  - person
  - bicycle
  - dog
```

## Quick Start

Run from the `sam3` repository root:

```bash
python auto_label_pipeline.py --src-dir src --output-dir runs/auto_label --overwrite
```

This command will:

- create the required input subfolders if missing
- remove the old output directory if `--overwrite` is set
- load prompts and inputs
- run inference
- write YOLO dataset outputs

## Recommended Commands

### Default run

```bash
python auto_label_pipeline.py \
  --src-dir src \
  --output-dir runs/auto_label \
  --overwrite
```

### Use a specific checkpoint

```bash
python auto_label_pipeline.py \
  --src-dir src \
  --output-dir runs/auto_label \
  --checkpoint checkpoints/sam3.pt \
  --overwrite
```

### Process videos at a lower frame rate

```bash
python auto_label_pipeline.py \
  --src-dir src \
  --output-dir runs/auto_label \
  --frame-fps 2 \
  --overwrite
```

### Run on CPU

```bash
python auto_label_pipeline.py \
  --src-dir src \
  --output-dir runs/auto_label \
  --device cpu \
  --overwrite
```

## Full CLI Reference

```text
--src-dir SRC_DIR
    Resource directory containing images/, videos/, and prompts.yaml.

--prompts-file PROMPTS_FILE
    Optional custom prompt YAML path. Defaults to <src-dir>/prompts.yaml.

--output-dir OUTPUT_DIR
    Output directory for extracted frames, dataset images, labels, and metadata.

--checkpoint CHECKPOINT
    Optional local SAM3 checkpoint path.

--checkpoint-url CHECKPOINT_URL
    Fallback download URL if no local checkpoint is available.

--checkpoint-cache-dir CHECKPOINT_CACHE_DIR
    Cache directory used for downloaded checkpoints. Default: checkpoints

--device DEVICE
    Inference device, usually cuda or cpu.

--threshold THRESHOLD
    SAM3 confidence threshold. Default: 0.5

--min-mask-area MIN_MASK_AREA
    Minimum mask area in pixels kept as a valid instance. Default: 64

--min-box-size MIN_BOX_SIZE
    Minimum width and height in pixels kept as a valid instance. Default: 4

--max-polygon-points MAX_POLYGON_POINTS
    Maximum polygon points per instance after contour simplification. Default: 200

--frame-fps FRAME_FPS
    Optional extraction FPS for videos. If omitted, all frames are extracted.

--overwrite
    Delete the old output directory before the run.
```

## Output Structure

After a successful run, the output directory looks like this:

```text
runs/auto_label/
  extracted_frames/
  dataset/
    images/
    labels/
  dataset.yaml
  manifest.json
```

### `extracted_frames/`

Contains video frames created by `ffmpeg`.

Example:

```text
extracted_frames/
  videos__bedroom/
    000001.jpg
    000002.jpg
```

### `dataset/images/`

Contains copied source images and copied extracted frames.

Examples:

- `images__test_image.jpg`
- `videos__bedroom__000001.jpg`

### `dataset/labels/`

Contains YOLO segmentation label files with one instance per line.

Example file content:

```text
0 0.387500 0.000000 0.348958 0.001852 ...
0 0.297917 0.442593 0.270833 0.440741 ...
```

Format notes:

- first value is the class id
- remaining values are normalized polygon coordinates
- coordinates are written as `x1 y1 x2 y2 ...`
- empty file means the image was processed but no valid instances were kept

### `dataset.yaml`

YOLO dataset definition.

Example:

```yaml
path: /absolute/path/to/runs/auto_label/dataset
train: images
val: images
test: images
names:
  0: person
  1: car
```

### `manifest.json`

Processing record with classes and per-file outputs.

It includes:

- original source path
- output image path
- output label path
- instance count per item

## End-to-End Example

The repository already contains sample assets:

- `src/images/test_image.jpg`
- `src/videos/bedroom.mp4`
- `src/prompts.yaml`

Example command used for validation:

```bash
python auto_label_pipeline.py \
  --src-dir src \
  --output-dir runs/auto_label_check \
  --frame-fps 1 \
  --overwrite
```

Observed result in this environment:

- 1 input image processed
- 1 input video extracted into 7 frames
- 8 total output images/frames written
- 7 video frame label files generated successfully

## Validation and Testing

### Run the pipeline smoke test manually

```bash
python auto_label_pipeline.py --src-dir src --output-dir runs/auto_label_check --frame-fps 1 --overwrite
```

### Run regression tests

```bash
scripts/run_tests.sh tests/test_auto_label_pipeline.py
```

The repository includes `scripts/run_tests.sh`, which disables third-party pytest plugin autoloading so local tests are less likely to break because of unrelated global environment plugins.

Current regression tests cover:

- single-channel 3D mask handling in polygon conversion
- correct output naming for extracted video frames

## Troubleshooting

### `No inputs found`

Cause:

- `src/images/` and `src/videos/` are both empty

Fix:

- add image or video files into the input folders

### `Prompt file not found`

Cause:

- `prompts.yaml` is missing

Fix:

- create `src/prompts.yaml` or pass `--prompts-file`

### checkpoint download or loading failure

Cause:

- no local checkpoint exists and remote download fails

Fix:

- place the model at `checkpoints/sam3.pt`
- or run with `--checkpoint /path/to/sam3.pt`
- or verify network access to the configured `--checkpoint-url`

### `ffmpeg` command fails

Cause:

- `ffmpeg` is not installed or the input video is invalid

Fix:

- install `ffmpeg`
- validate the video with `ffprobe`

### empty label files

Cause:

- the prompt did not match anything confidently enough
- detections were filtered by threshold, mask area, or box size

Fix:

- improve prompts in `prompts.yaml`
- try lowering `--threshold`
- lower `--min-mask-area`
- lower `--min-box-size`

### output contains only one class repeatedly

Cause:

- prompts may be too broad, too narrow, or overlapping
- one class may dominate SAM3 text matching

Fix:

- rewrite prompts to be more specific
- inspect a few output labels manually
- compare per-class results across several frames before large batch runs

## Practical Tuning Advice

- Start with 2 to 5 classes, not dozens at once.
- Prefer concrete prompts like `white sedan` or `safety helmet` over vague prompts like `object`.
- For long videos, use `--frame-fps` to control output size.
- If polygons are too detailed, reduce `--max-polygon-points`.
- If small objects disappear, reduce `--min-mask-area` and `--min-box-size`.
- If false positives are high, increase `--threshold`.

## Known Behavior

- Images and video frames are flattened into filename-safe outputs using `__` separators.
- Video frames are extracted first, then copied into the YOLO dataset image directory.
- Duplicate masks across classes are suppressed when their mask IoU is very high.
- Empty label files are valid outputs for images with no kept instances.

## Suggested Workflow

1. put a small sample of images and videos into `src/`
2. write 2 to 5 classes in `src/prompts.yaml`
3. run the pipeline into a disposable output directory
4. inspect several labels in `dataset/labels/`
5. adjust prompts and thresholds
6. rerun at larger scale

## Important Paths

- script: `auto_label_pipeline.py`
- prompts: `src/prompts.yaml`
- sample input image: `src/images/test_image.jpg`
- sample input video: `src/videos/bedroom.mp4`
- default checkpoint cache: `checkpoints/sam3.pt`
- tests: `tests/test_auto_label_pipeline.py`
- test runner: `scripts/run_tests.sh`

## Changelog Notes for This Repository Copy

The current repository state has been validated with:

- bundled sample image input
- bundled sample video input
- working video frame output naming
- working 3D mask to polygon conversion
- targeted regression tests for both fixes
