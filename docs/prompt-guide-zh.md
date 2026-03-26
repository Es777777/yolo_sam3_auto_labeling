# Prompt 配置指南

这份文档专门说明如何为 `auto_label_pipeline.py` 配置 `src/prompts.yaml`，让自动标注结果更稳定、更接近你的真实目标。

## 先记住一个原则

推荐默认使用英文 prompt。

原因：

- 对这类视觉文本提示模型来说，英文通常更稳定
- 英文更适合描述目标、背景、位置和外观限制
- 多类别一起跑时，英文 prompt 往往更容易拉开类别边界

最推荐的做法是：

- `name` 可以用你希望导出的类别名
- `prompt` 尽量写英文

例如：

```yaml
classes:
  - name: 人
    prompt: person in a bedroom
  - name: 床
    prompt: bed in an indoor bedroom scene
  - name: 枕头
    prompt: pillow on the bed
```

这里：

- `name` 决定导出到 `dataset.yaml` 的类别名称
- `prompt` 决定模型实际去找什么目标

## prompts.yaml 的基本格式

文件路径：`src/prompts.yaml`

基本结构：

```yaml
classes:
  - name: person
    prompt: person
  - name: bed
    prompt: bed
  - name: pillow
    prompt: pillow
```

你也可以写成简写形式：

```yaml
classes:
  - person
  - bed
  - pillow
```

但更推荐完整写法，因为后面更容易调优。

## `name` 和 `prompt` 的区别

### `name`

`name` 是输出类别名，主要影响：

- `dataset.yaml`
- `manifest.json`
- 训练时看到的类别名称

### `prompt`

`prompt` 是给模型的文字提示，直接影响：

- 模型是否能找到目标
- 类别是否容易混淆
- 检测是否稳定

所以配置时最重要的是 `prompt`，不是 `name`。

## prompt 应该怎么写

推荐按这个思路来写：

```text
目标 + 场景/背景 + 位置关系 + 外观限制
```

不一定每条都要写满，但这是一个很好用的模板。

例如：

- `person`
- `person in a bedroom`
- `pillow on the bed`
- `table lamp beside the bed`
- `forklift in a warehouse`
- `traffic light at an intersection`

## 如何表达“背景、目标、场景”

如果你希望在启动前就把“背景”“目标”“场景”说清楚，当前版本最直接的方法，就是把这些信息写进每个类的 `prompt`。

例如，你的数据是卧室场景：

```yaml
classes:
  - name: person
    prompt: person in an indoor bedroom scene
  - name: bed
    prompt: bed in an indoor bedroom scene
  - name: lamp
    prompt: lamp near the bed in a bedroom
```

如果是仓库场景：

```yaml
classes:
  - name: forklift
    prompt: forklift in a warehouse
  - name: pallet
    prompt: wooden pallet on the warehouse floor
  - name: worker
    prompt: warehouse worker wearing a safety vest
```

如果是道路场景：

```yaml
classes:
  - name: car
    prompt: car on an urban street
  - name: bus
    prompt: bus on a city road
  - name: pedestrian
    prompt: pedestrian crossing the road
```

也就是说，你完全可以通过 prompt 本身表达：

- 目标是什么
- 出现在什么背景里
- 在什么位置附近
- 有什么外观特征

## 好 prompt 和差 prompt 的区别

### 好 prompt 的特点

- 具体
- 可视化强
- 跟真实场景一致
- 和其他类别边界清楚

好例子：

- `pillow on the bed`
- `bedside table`
- `worker wearing a yellow helmet`
- `red forklift in a warehouse`

### 差 prompt 的常见问题

太宽泛：

- `object`
- `equipment`
- `item`

太抽象：

- `danger`
- `work area`
- `rest zone`

太重叠：

- `car`
- `vehicle`
- `sedan`

这些会导致：

- 噪声多
- 类别互相抢结果
- 掩码重复或不稳定

## 中文和英文怎么选

### 最推荐

```yaml
classes:
  - name: person
    prompt: person in a bedroom
```

### 也推荐

```yaml
classes:
  - name: 人
    prompt: person in a bedroom
```

### 不太推荐

```yaml
classes:
  - name: 人
    prompt: 卧室里的人
```

如果你想兼顾“导出中文类名”和“模型效果稳定”，最佳组合就是：

- `name` 用中文
- `prompt` 用英文

## 类别粒度怎么定

先想清楚你到底要“粗分类”还是“细分类”。

### 粗分类

如果你只关心大类：

```yaml
classes:
  - name: person
    prompt: person
  - name: car
    prompt: car
```

### 细分类

如果你想区分更细的目标：

```yaml
classes:
  - name: sedan
    prompt: sedan car on a street
  - name: delivery_van
    prompt: delivery van on a street
```

注意不要在同一次运行里放太多高度重叠的类，除非你已经验证过效果。

## 针对不同场景的模板

### 室内房间场景

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

### 工厂或仓库场景

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

### 路面交通场景

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

## 当前仓库样例怎么配更合适

这个仓库自带的视频是卧室场景，所以默认更适合：

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

相比原来用 `car`，这组配置更符合样例内容。

## 调 prompt 的推荐流程

推荐按这个流程做：

1. 先只写 2 到 5 个类别
2. 每个类别先用英文基础 prompt
3. 跑几张图或一小段视频
4. 检查输出标签是否合理
5. 如果误检多，就把 prompt 写得更具体
6. 如果漏检多，就把 prompt 写得更自然、更常见
7. 稳定后再扩大数据规模

## 常见调优思路

### 误检太多

可以尝试：

- 提高 `--threshold`
- prompt 写得更具体
- 加上场景限制

例如把：

- `lamp`

改成：

- `table lamp beside the bed`

### 漏检太多

可以尝试：

- prompt 写得更简单
- 去掉太强的限制词
- 降低 `--threshold`

例如把：

- `large warm-colored modern bedside lamp with visible stand`

改成：

- `bedside lamp`

### 类别之间互相混

可以尝试：

- 避免同一轮里同时放高度相似的类别
- 给每个类加一点区分场景或结构描述

例如不要同时直接写：

- `car`
- `vehicle`
- `sedan`

## 一个实用的配置模板

如果你暂时不知道怎么写，可以先从这个模板改：

```yaml
classes:
  - name: class_1
    prompt: main object in the target scene
  - name: class_2
    prompt: second object in the target scene
  - name: class_3
    prompt: third object near the main area
```

然后把它替换成你的实际目标。

## 跑之前的检查清单

在运行 pipeline 之前，检查一下：

- 每个类是不是明确的可见物体
- `prompt` 是否尽量用了英文
- prompt 是否和真实场景匹配
- 类别之间是否过于重叠
- 你是否先用少量样本试跑过

## 相关文件

- Prompt 配置文件：`src/prompts.yaml`
- 主使用指南：`pipeline.md`
- 测试脚本：`scripts/run_tests.sh`

如果后面你有真实业务场景，我也可以直接按照你的场景帮你生成一版可用的 `src/prompts.yaml`。
