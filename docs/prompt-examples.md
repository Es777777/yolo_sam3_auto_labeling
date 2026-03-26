# Prompt Examples Library

This file provides reusable prompt templates for common scenes. You can copy one block into `src/prompts.yaml` and then adjust it for your own data.

General recommendation:

- keep `prompt` in English
- keep classes concrete and visual
- start small, then expand
- avoid heavily overlapping classes in the same run unless needed

## Indoor Bedroom

Good for the bundled sample video in this repository.

```yaml
classes:
  - name: person
    prompt: person in a bedroom
  - name: bed
    prompt: bed in a bedroom
  - name: pillow
    prompt: pillow on the bed
  - name: lamp
    prompt: lamp beside the bed
```

More specific variant:

```yaml
classes:
  - name: person
    prompt: person in an indoor bedroom scene
  - name: bed
    prompt: bed in an indoor bedroom scene
  - name: pillow
    prompt: pillow on the bed
  - name: bedside_lamp
    prompt: bedside lamp near the bed
```

## Living Room

```yaml
classes:
  - name: person
    prompt: person in a living room
  - name: sofa
    prompt: sofa in a living room
  - name: coffee_table
    prompt: coffee table in front of a sofa
  - name: tv
    prompt: television in a living room
```

## Office Scene

```yaml
classes:
  - name: person
    prompt: office worker
  - name: desk
    prompt: office desk
  - name: chair
    prompt: office chair
  - name: laptop
    prompt: laptop on a desk
  - name: monitor
    prompt: computer monitor on a desk
```

## Warehouse

```yaml
classes:
  - name: worker
    prompt: warehouse worker
  - name: forklift
    prompt: forklift in a warehouse
  - name: pallet
    prompt: pallet on the warehouse floor
  - name: shelf
    prompt: storage shelf in a warehouse
```

Safer version with stronger constraints:

```yaml
classes:
  - name: worker
    prompt: warehouse worker wearing a safety vest
  - name: forklift
    prompt: red or yellow forklift in a warehouse
  - name: pallet
    prompt: wooden pallet on the floor
  - name: carton_box
    prompt: cardboard box on a pallet
```

## Factory Workshop

```yaml
classes:
  - name: worker
    prompt: factory worker
  - name: helmet
    prompt: safety helmet
  - name: machine
    prompt: industrial machine in a factory
  - name: conveyor
    prompt: conveyor belt in a factory
```

## Construction Site

```yaml
classes:
  - name: worker
    prompt: construction worker
  - name: helmet
    prompt: safety helmet on a worker
  - name: excavator
    prompt: excavator on a construction site
  - name: truck
    prompt: construction truck on a site
```

## Street Traffic

```yaml
classes:
  - name: car
    prompt: car on an urban street
  - name: bus
    prompt: bus on a city street
  - name: bicycle
    prompt: bicycle on the road
  - name: pedestrian
    prompt: pedestrian on a sidewalk
  - name: traffic_light
    prompt: traffic light at an intersection
```

## Parking Lot

```yaml
classes:
  - name: car
    prompt: parked car in a parking lot
  - name: suv
    prompt: suv in a parking lot
  - name: person
    prompt: person walking in a parking lot
  - name: parking_line
    prompt: parking lane marking on the ground
```

Note: `car` and `suv` may overlap. If results become unstable, keep only one of them.

## Retail Store

```yaml
classes:
  - name: person
    prompt: customer in a store
  - name: shelf
    prompt: product shelf in a retail store
  - name: shopping_cart
    prompt: shopping cart in a store
  - name: checkout_counter
    prompt: checkout counter in a store
```

## Restaurant or Cafe

```yaml
classes:
  - name: person
    prompt: customer in a cafe
  - name: table
    prompt: dining table in a cafe
  - name: chair
    prompt: dining chair in a cafe
  - name: cup
    prompt: cup on a table
```

## Classroom

```yaml
classes:
  - name: student
    prompt: student in a classroom
  - name: desk
    prompt: student desk in a classroom
  - name: chair
    prompt: classroom chair
  - name: whiteboard
    prompt: whiteboard in a classroom
```

## Roadside Safety or Surveillance

```yaml
classes:
  - name: person
    prompt: person near a road
  - name: car
    prompt: car on a road
  - name: motorcycle
    prompt: motorcycle on a road
  - name: bicycle
    prompt: bicycle on a road
  - name: traffic_cone
    prompt: traffic cone on the roadside
```

## Farm or Outdoor Field

```yaml
classes:
  - name: person
    prompt: person in a field
  - name: tractor
    prompt: tractor in a farm field
  - name: animal
    prompt: farm animal in a field
  - name: fence
    prompt: fence in a farm area
```

## PPE Detection

This type of task often works better when prompts are concrete and tied to visible items.

```yaml
classes:
  - name: worker
    prompt: worker in an industrial area
  - name: helmet
    prompt: safety helmet on a person
  - name: safety_vest
    prompt: reflective safety vest worn by a worker
  - name: gloves
    prompt: protective gloves on hands
```

## Tips for Adapting a Template

- Replace scene words to match your real environment.
- Remove classes you do not actually need.
- Start with 3 to 5 classes, not 15 at once.
- If false positives are high, make prompts more specific.
- If recall is low, simplify the prompt.

## Suggested Testing Loop

1. copy one template into `src/prompts.yaml`
2. run a small sample with `auto_label_pipeline.py`
3. inspect `runs/.../dataset/labels/`
4. refine prompts and rerun

## Related Files

- main pipeline guide: `pipeline.md`
- Chinese prompt guide: `prompt-guide-zh.md`
- active prompt config: `src/prompts.yaml`
