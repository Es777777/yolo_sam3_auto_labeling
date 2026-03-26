# src Resource Folder

Put all labeling resources here before running the pipeline.

- `images/`: images to label directly.
- `videos/`: videos to split into frames, then label frame by frame.
- `prompts.yaml`: text prompts and class definitions.

Example:

```yaml
classes:
  - name: person
    prompt: person
  - name: white_car
    prompt: white car
  - name: traffic_light
    prompt: traffic light
```

Notes:

- `name` becomes the YOLO class name.
- `prompt` is the SAM3 text prompt used for auto labeling.
- Videos are always extracted into single frames before inference.
