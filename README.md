# NavCorrector Demo

这是一个面向后续开发的最小 demo 基座，核心只保留两条主链：

- `train.py`：训练 `posenet` 和 `navigator`
- `test.py`：测试 `posenet + navigator`
- `posenet_test.py`：单独测试 `posenet`

数据目录 `RIDI/`、`OXIOD/`、`RONIN/`、`SELFMADE/` 仍保留，但当前主链只直接使用 `RIDI/` 和 `OXIOD/`。

## 目录结构

核心文件：

- `train.py`
- `test.py`
- `posenet_test.py`
- `models/posenet.py`
- `models/navigator.py`
- `data/dataset_RIDI.py`
- `data/dataset_OXIOD.py`
- `utils/training_utils.py`
- `utils/visualization.py`
- `src/pdr.py`

归档实验脚本：

- 根目录里不再保留历史实验入口
- 旧的量化 / ECC / 在线基线等实验脚本统一移动到 `archive/root_experiments/`

输出目录：

- `checkpoints/<dataset>/`
- `output/posenet/<dataset>/`
- `output/posenet+navigator/<dataset>/`

其中 `<dataset>` 目前支持：

- `ridi`
- `oxiod`

## 当前架构

### 1. posenet

`models/posenet.py`

作用：

- 输入一个 IMU 窗口 `[B, T, 6]`
- 预测窗口对应的相对旋转四元数

当前实现：

- Transformer 编码器
- 输出四元数
- 训练目标是相对姿态监督

训练入口：

- `train.py` 里的 `train_posenet(...)`

测试入口：

- `posenet_test.py`
- `test.py` 在 `navigator` 推理时也会调用它

### 2. navigator

`models/navigator.py`

作用：

- 输入 IMU 窗口
- 预测窗口对应的位移增量 `dp_body`

当前实现：

- 1D CNN / ResNet 风格特征提取
- MLP 回归 3D 位移

训练入口：

- `train.py` 里的 `train_navigator(...)`

测试入口：

- `test.py`

### 3. 数据层

`data/dataset_RIDI.py` 和 `data/dataset_OXIOD.py`

作用：

- 读取原始数据
- 做窗口切分
- 生成：
  - IMU 输入
  - 相对姿态标签
  - body/world 位移标签

`utils/training_utils.py`

作用：

- 把数据集窗口进一步整理成 `train.py` 直接可用的 Tensor
- 提供：
  - `load_data_ridi_absheading(...)`
  - `load_data_oxiod_absheading(...)`

### 4. 可视化层

`utils/visualization.py`

作用：

- 集中存放所有测试脚本使用的绘图函数

当前被：

- `test.py`
- `posenet_test.py`

共同使用

### 5. PDR

`src/pdr.py`

作用：

- 提供独立的 PDR 基线逻辑
- 后续可以用于和模型结果做对比

当前主链没有直接调用它，但这是保留文件之一。

## 训练流程

`train.py` 分三步：

1. 训练 `posenet`
2. 冻结 `posenet`，训练 `navigator`
3. 联合微调 `posenet + navigator`

训练产物会写到：

- `checkpoints/<dataset>/posenet.pth`
- `checkpoints/<dataset>/navigator.pth`

## 测试流程

### 1. 只测试 posenet

运行：

```bash
python posenet_test.py
```

可选切换数据集：

```bash
DATASET=OXIOD python posenet_test.py
```

输出目录：

```text
output/posenet/<dataset>/
```

主要输出：

- 相对旋转图
- 轨迹重建图
- 航向分析图
- 绝对姿态图

### 2. 测试 posenet + navigator

运行：

```bash
python test.py
```

可选切换数据集：

```bash
DATASET=OXIOD python test.py
```

输出目录：

```text
output/posenet+navigator/<dataset>/
```

主要输出：

- 2D 轨迹对比
- 累积位移图
- 每窗口 world 位移图
- 每窗口 body 位移图

## 如何训练

默认训练 RIDI：

```bash
python train.py
```

训练 OXIOD：

```bash
DATASET=OXIOD python train.py
```

## 依赖

当前主链至少依赖：

- `torch`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `tqdm`

如果这些库缺失，训练或测试脚本会在 import 阶段失败。

## 约定

- 所有路径都基于脚本所在目录推导，不依赖绝对路径
- 所有 checkpoint 统一写到 `checkpoints/`
- 所有测试输出统一写到 `output/`
- `posenet` 表示姿态网络
- `navigator` 表示位移网络

## 推荐运行顺序

1. 准备 `RIDI/` 或 `OXIOD/` 数据
2. 运行 `python train.py`
3. 运行 `python posenet_test.py`
4. 运行 `python test.py`
5. 需要基线对比时，再接入 `src/pdr.py`
