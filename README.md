# Animal Recognition

基于 PyTorch、Torchvision 和 Gradio 的猫狗品种识别项目。模型使用 EfficientNet-B0，对 Oxford-IIIT Pet Dataset 中的 37 个猫狗品种进行分类，并支持对不在训练范围内的图片进行低置信度拒识。

## 功能

- 37 类猫狗品种分类
- Gradio 网页界面
- 命令行单图预测
- 训练、评估与结果可视化
- 仓库内附最佳模型权重和示例图片

## 环境要求

- Python 3.10+
- 建议使用支持 CUDA 的 NVIDIA GPU 进行训练；预测可使用 CPU

安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行网页应用

```powershell
python app.py
```

浏览器访问 `http://127.0.0.1:7860`，上传图片后即可查看预测结果。

## 命令行预测

```powershell
python predict.py photo/Beagle.jpg --top_k 3
```

默认加载 `output/models/best_model.pth`。

## 准备数据和训练

训练数据不会提交到 Git 仓库。运行以下命令可下载或整理数据集：

```powershell
python setup_data.py
python train.py
```

数据目录格式：

```text
data/
  train/
    Abyssinian/
    ...
  val/
    Abyssinian/
    ...
```

## 项目结构

```text
app.py                 Gradio 应用
config.py              项目配置
data_utils.py          数据加载与增强
model.py               模型定义
predict.py             推理工具
setup_data.py          数据准备
train.py               训练入口
generate_figures.py    结果图生成
photo/                 示例图片
output/figures/        训练与评估图表
output/models/         最佳模型权重
```

## 数据与模型

数据来源为 [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)。完整训练集可通过 `setup_data.py` 重新获取，因此不纳入版本控制。仓库只保留部署所需的最佳模型，避免重复检查点造成仓库膨胀。
