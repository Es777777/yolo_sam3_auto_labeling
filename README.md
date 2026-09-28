# SAM3 Text Auto Label Pipeline

## Local Desktop App

Run the native desktop labeling tool:

```bash
bash start.sh
```

Choose an image folder or ZIP archive, enter one text target per line, run SAM3, then review boxes and polygons in the built-in viewer. The original images are not modified. Results preserve relative paths under `images/`, `labels_detect/`, and `labels_segment/`.

## 新版文本标注工作台

现在可通过浏览器输入文本、批量标注、人工复核并导出 YOLO 检测或分割数据集。
在上一级目录运行 `bash start.sh`，打开 http://127.0.0.1:7860 。
完整中文说明见 [../README.md](../README.md)。新版入口为 `label_tool.py`；以下内容为旧命令行流程。

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

## Desktop Quick Start

1. Create an environment and install dependencies.

```bash
conda create -n sam3-auto-label python=3.12
conda activate sam3-auto-label
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -e ".[dev]"
```

2. Start the desktop tool.

```bash
bash start.sh
```

Enter prompts like:

```text
人 = person
汽车 = car
安全帽 = safety helmet | hard hat
```

The older command line pipeline remains available:

```bash
python auto_label_pipeline.py --src-dir src --output-dir runs/auto_label --overwrite
```

## CUDA and CPU

The desktop app uses CUDA automatically when available. Set `SAM3_PYTHON=/path/to/python` to select another environment. Place the local checkpoint at `checkpoints/sam3.pt`; it is intentionally ignored by Git.

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
- `test_inputs/` contains lightweight sample inputs for validation; personal image collections under `src/` are ignored
- the repository keeps the SAM3 package code needed by the pipeline, while the project presentation is focused on the auto-labeling workflow

## License

This repository includes upstream SAM3 code under the original license in `LICENSE`.
