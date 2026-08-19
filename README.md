# PSDG-Net 开源方法包

这是论文 `PSDG-Net: typed physical-semantic graphs for generative multidisciplinary engineering design` 的方法实现与两个异构任务复现实验。代码和目录只保留论文主方法、机器人机电执行器实验、热-流-固实验及测试所需内容。

## 包含内容

- `psdg_net/`：可执行物理语义图方法。
  - `schema.py`：变量、单位维度、节点、端口、方程、约束和关系定义。
  - `compiler.py`：变量归一化、节点契约构建、因果掩码、关系证据和四步全协方差传播。
  - `model.py`：关系类型专属值算子、物理契约注意力、三层 PSDG-Net 多任务预测器。
  - `generator.py`：12 维潜变量的图条件 VAE、重建+KL 损失、边界投影和 20 步预测器引导优化。
  - `active_learning.py`：不确定性、可行性边界、改进、分集、物理残差和验证成本六项采集接口。
- `experiments/`：机器人机电与热-流-固两个实验的 schema、物理 provider、Sobol 数据生成和统一训练器。
- `data/robot_electromechanical/`、`data/thermal_fluid_mechanical/`：两个异构实验的 32,768 个候选数据、固定 train/validation/test 划分及 schema 元数据。
- `tests/`：方法前向传播、协方差传播、主动学习、生成器和两个实验数据契约测试。

飞行主任务数据没有放入本目录；论文主方法仍可通过任何符合 `PhysicsFeatureProvider` 接口的任务 provider 使用。对比模型、消融实验、飞行任务数据、训练 checkpoint、预测结果和调试分支均未包含。

## 安装

```powershell
cd open_source_main_method
python -m pip install -e .
```

依赖为 Python 3.8+、NumPy、SciPy、scikit-learn 和 PyTorch。CPU 与 CUDA 均可运行；大规模训练建议使用 CUDA。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 运行两个复现实验

默认读取仓库内已提供的数据，不会读取项目外部代码或数据：

```powershell
python -m experiments.train --domain robot --device cpu
python -m experiments.train --domain thermal --device cpu
```

默认使用论文异构实验配置：hidden width 128、4 个 attention heads、3 个关系感知 block、dropout 0.02、AdamW、学习率 `4e-4`、batch size 1024、最多 40 epochs、`log1p` 目标变换。训练输出写入 `runs/`，不会覆盖 `data/`。

如果希望从代码重新生成 Sobol 数据，可直接调用：

```python
from experiments.robot_actuator import build_schema, generate_dataset

schema = build_schema()
dataset = generate_dataset(schema, n_power=15, seed=2027)
```

## 方法使用骨架

```python
import torch
from psdg_net import PSDGNet

model = PSDGNet(schema, out_dim=number_of_targets, hidden=128, heads=4, layers=3)
physics = provider.evaluate(x_numpy).torch(torch.device("cpu"))
prediction, feasibility_logit, graph = model(torch.as_tensor(x_numpy), physics=physics)
```

provider 应返回 `PhysicsBatch`，为每个样本提供 `input_std`、`node_extras`、`equation_residuals`、`constraint_margins`、`relation_strength` 和 `fidelity`。这些是图契约证据，不是目标预测、目标残差或低保真标签旁路。

## 与论文方法的对应关系

标量变量按边界归一化并裁剪到 `[-4, 4]`；局部不确定性通过四步固定点递推传播到完整因果图；注意力先用 schema 因果掩码移除未声明传递，再融合耦合强度、方程残差、fidelity 和传播不确定性；回归 head 直接预测每个目标，另设可行性 logistic head。生成器固定 context/safety 变量，只解码 design 变量，输出再投影回物理边界。

## 许可

见 [LICENSE](LICENSE)。
