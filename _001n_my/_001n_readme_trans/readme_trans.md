<div align="center">
<img src="https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images/libero_logo.png" width="360">

<p align="center">
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/actions">
<img alt="测试通过" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/graphs/contributors">
<img alt="GitHub 贡献者" src="https://img.shields.io/github/contributors/Lifelong-Robot-Learning/LIBERO" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/issues">
<img alt="问题" src="https://img.shields.io/github/issues/Lifelong-Robot-Learning/LIBERO?color=0088ff" />
</a>
</p>

## **终身机器人学习中知识迁移的基准评测**

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone

[[项目网站]](https://libero-project.github.io)
[[论文]](https://arxiv.org/pdf/2306.03310.pdf)
[[文档]](https://lifelong-robot-learning.github.io/LIBERO/)
______________________________________________________________________
![概览图](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images//fig1.png)
</div>

**LIBERO** 旨在研究多任务机器人学习和终身机器人学习问题中的知识迁移。成功解决这些问题，既需要关于物体和空间关系的声明性知识，也需要关于运动和行为的程序性知识。**LIBERO** 提供：

- 一套程序化生成流程，原则上能够生成无限多个操作任务。
- 130 个任务，分为四个任务套件：**LIBERO-Spatial**、**LIBERO-Object**、**LIBERO-Goal** 和 **LIBERO-100**。前三个任务套件具有受控的分布偏移，也就是说，它们要求迁移某一种特定类型的知识。相比之下，**LIBERO-100** 包含 100 个操作任务，需要迁移相互交织的多种知识。**LIBERO-100** 进一步划分为用于策略预训练的 **LIBERO-90**，以及用于测试智能体下游终身学习性能的 **LIBERO-10**。
- 五个研究主题。
- 三种视觉运动策略网络架构。
- 三种终身学习算法，以及顺序微调和多任务学习基线。

---

# 目录

- [安装](#安装)
- [数据集](#数据集)
- [快速入门](#快速入门)
  - [任务](#任务)
  - [训练](#训练)
  - [评估](#评估)
- [引用](#引用)
- [许可证](#许可证)

# 安装

请按以下顺序运行命令，安装 **LIBERO** 的依赖项。

```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

然后安装 `libero` 包：

```
pip install -e .
```

# 数据集

我们为 **LIBERO** 中的四个任务套件提供了高质量的人工遥操作示范。要下载示范数据集，请运行：

```python
python benchmark_scripts/download_libero_datasets.py
```

默认情况下，数据集将存储在 `LIBERO` 文件夹下，并下载全部四个数据集。要下载某个特定数据集，请使用：

```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```

其中，`DATASET` 从 `[libero_spatial, libero_object, libero_100, libero_goal]` 中选择。

**新增功能！！！**

你也可以使用以下命令从 HuggingFace 下载数据集：

```python
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

此选项也可以与指定数据集的选项结合使用：

```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

托管在 HuggingFace 上的数据集可在[此处](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)获取。

# 快速入门

如需详细的操作指南，请参阅文档，或查看 `notebooks` 文件夹中的笔记本示例。下面提供了获取任务、训练和评估的示例脚本。

## 任务

下面是从指定任务套件中获取某个具体任务的最小示例。

```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # 也可以选择 libero_spatial、libero_object 等。
task_suite = benchmark_dict[task_suite_name]()

# 获取某个具体任务
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# 在环境中逐步执行动作
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # 为进行基准评测，我们固定使用一组初始状态。
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
env.close()
```

目前，我们仅支持稀疏奖励函数，即智能体在完成任务时获得 `+1` 的奖励。由于稀疏奖励下的强化学习极难学好，目前我们主要关注终身模仿学习。

## 训练

要开始一项终身学习实验，请选择：

- `BENCHMARK`：从 `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]` 中选择。
- `POLICY`：从 `[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy]` 中选择。
- `ALGO`：从 `[base, er, ewc, packnet, multitask]` 中选择。

然后运行以下命令：

```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python libero/lifelong/main.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO
```

有关复现研究结果的详细信息，请参阅文档。

## 评估

默认情况下，策略会在训练过程中即时进行评估。如果你的图形处理器（GPU）计算资源有限，我们提供了一个评估脚本，供你单独评估模型。

```shell
python libero/lifelong/evaluate.py --benchmark BENCHMARK_NAME \
                                   --task_id TASK_ID \ 
                                   --algo ALGO_NAME \
                                   --policy POLICY_NAME \
                                   --seed SEED \
                                   --ep EPOCH \
                                   --load_task LOAD_TASK \
                                   --device_id CUDA_ID
```

# 引用

如果你觉得 **LIBERO** 对你的研究有帮助，请考虑引用我们的论文：

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

# 许可证

| 组成部分 | 许可证 |
|----------|--------|
| 代码库 | [MIT 许可证](../../LICENSE) |
| 数据集 | [知识共享署名 4.0 国际许可协议（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/legalcode) |
