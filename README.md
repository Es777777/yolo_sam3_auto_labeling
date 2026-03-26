# SAM3 Auto Label Pipeline

A personal project for turning image and video collections into YOLO segmentation datasets with SAM3 text prompts.

This repo is focused on one practical workflow: prepare a small prompt file, run inference on images or videos, and export labels that can be used directly for training or inspection.

## Highlights

- one pipeline for both images and videos
- YOLO segmentation export with `dataset.yaml` and `manifest.json`
- CUDA and CPU execution support
- prompt templates and a Chinese prompt guide
- lightweight sample inputs and regression tests

## Use Cases

- bootstrap labels for a custom dataset
- test prompt quality on a small batch before large-scale runs
- convert raw images or videos into YOLO-style segmentation training data

## How It Works

Given a resource folder with images, videos, and `prompts.yaml`, the pipeline will:

1. read class names and prompts
2. process images directly
3. extract frames from videos with `ffmpeg`
4. run SAM3 prompt-based segmentation
5. convert masks to YOLO polygons
6. export labels, dataset metadata, and a manifest

The main entrypoint is `auto_label_pipeline.py`.

## Repository Layout

```text
.
├── auto_label_pipeline.py      # main labeling script
├── sam3/                       # SAM3 package code used by the pipeline
├── src/                        # active local input folder
├── test_inputs/                # small sample inputs for smoke tests
├── tests/                      # regression tests
├── scripts/                    # helper scripts and test runner
├── docs/                       # usage and prompt-writing guides
└── pyproject.toml              # project metadata and dependencies
```

## Quick Start

1. Create an environment and install dependencies.

```bash
conda create -n sam3-auto-label python=3.12
conda activate sam3-auto-label
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -e ".[dev]"
```

2. Prepare inputs in `src/`.

- `src/images/`: images to label
- `src/videos/`: videos to split into frames and label
- `src/prompts.yaml`: class names and English prompts

Example prompt file:

```yaml
classes:
  - name: person
    prompt: person in an indoor room
  - name: chair
    prompt: chair in an indoor room
```

3. Run the pipeline.

```bash
python auto_label_pipeline.py --src-dir src --output-dir runs/auto_label --overwrite
```

4. Inspect outputs.

- `runs/auto_label/dataset/images`
- `runs/auto_label/dataset/labels`
- `runs/auto_label/dataset.yaml`
- `runs/auto_label/manifest.json`

## CUDA Validation

The pipeline has been validated locally on CUDA with:

```bash
python auto_label_pipeline.py --src-dir src --output-dir runs/auto_label_cuda_try --frame-fps 1 --overwrite
```

## Testing

Run the regression tests with:

```bash
scripts/run_tests.sh tests/test_auto_label_pipeline.py
```

## Documentation

- `docs/pipeline.md`: end-to-end usage guide and CLI reference
- `docs/prompt-guide-zh.md`: Chinese prompt writing guide
- `docs/prompt-examples.md`: reusable prompt templates

## GitHub Upload Notes

- `checkpoints/` is ignored and should stay local
- `runs/` is ignored and should stay local
- `src/` and `test_inputs/` contain lightweight sample inputs for validation
- the repository keeps the SAM3 package code needed by the pipeline, while the project presentation is focused on the auto-labeling workflow

## License

This repository includes upstream SAM3 code under the original license in `LICENSE`.
