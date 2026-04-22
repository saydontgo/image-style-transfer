# Image Style Transfer Coursework

本项目为：

1. 输入自己的照片或喜欢的照片。
2. 至少实现两种艺术风格迁移。
3. 训练自己的风格迁移模型。
4. 和网上下载的已训练模型做可复现实验对比。

## 1. 方法选择

项目采用 `Johnson et al., Perceptual Losses for Real-Time Style Transfer` 的快速风格迁移方案

## 2. 项目结构

```text
image-style-transfer/
├── train.py
├── stylize.py
├── compare_models.py
├── style_transfer/
│   ├── models/
│   │   ├── transformer_net.py
│   │   └── loss_network.py
│   └── utils/
│       └── image.py
├── checkpoints/
├── outputs/
├── external_models/
├── data/
│   ├── content_examples/
│   └── style_images/
└── docs/
    └── report_template.md
```

## 3. 数据准备

- 内容数据集：`MS-COCO 2014 train` 或 `MS-COCO 2017 train`
- 风格图：任选两张艺术作品，例如 `mosaic.jpg`、`candy.jpg`、`rain_princess.jpg`
- 测试内容图：放到 `data/content_examples/`

推荐目录：

```text
data/
├── coco_train2014/
│   ├── COCO_train2014_000000000009.jpg
│   ├── ...
├── content_examples/
│   ├── photo1.jpg
│   └── photo2.jpg
└── style_images/
    ├── mosaic.jpg
    └── candy.jpg
```

## 4. 推荐下载的公开预训练模型

优先推荐下载与本项目架构兼容的 PyTorch 权重：

- `pytorch/examples/fast_neural_style`
  - GitHub: <https://github.com/pytorch/examples/tree/main/fast_neural_style>
  - 这是最常见的 Johnson 快速风格迁移实现来源。
- `gordicaleksa/pytorch-neural-style-transfer-johnson`
  - GitHub: <https://github.com/gordicaleksa/pytorch-neural-style-transfer-johnson>
  - 仓库提供 MS-COCO 训练说明与预训练模型下载脚本。

建议优先下载以下风格的 `.pth` 权重到 `external_models/`：

- `mosaic.pth`
- `candy.pth`
- `rain_princess.pth`
- `udnie.pth`

说明：

- 本项目可以直接加载以下三类 Johnson 风格迁移权重：
  - 本仓库自己训练导出的 `.pth` / `.ckpt`
  - `pytorch/examples/fast_neural_style` 导出的 PyTorch 权重
  - `gordicaleksa/pytorch-neural-style-transfer-johnson` 导出的 PyTorch 权重
- 如果下载的是 `.ckpt`，也可以直接加载。
- 如果下载的是 `ONNX` 文件，则不能直接用本项目的 PyTorch 推理脚本，需要单独写 ONNXRuntime 推理代码。
- 如果报错里出现类似 `conv2.conv1.weight`、`upconv1.conv1.conv1.weight` 这类 key，通常说明该权重来自另一套网络实现，不是本项目支持的 Johnson `TransformerNet`。

## 5. 训练自己的模型

以下示例使用 `mosaic` 风格训练一个模型：

```bash
python train.py \
  --dataset data/coco_train2014 \
  --style-image data/style_images/mosaic.jpg \
  --preview-dir data/content_examples \
  --output-dir checkpoints \
  --run-name mosaic_custom \
  --image-size 256 \
  --style-size 512 \
  --batch-size 8 \
  --epochs 2 \
  --subset-size 20000 \
  --learning-rate 2e-4 \
  --style-weight 1e5 \
  --content-weight 1.0 \
  --tv-weight 1e-6 \
  --mixed-precision
```

训练第二种风格时只要换风格图和运行名：

```bash
python train.py \
  --dataset data/coco_train2014 \
  --style-image data/style_images/candy.jpg \
  --preview-dir data/content_examples \
  --output-dir checkpoints \
  --run-name candy_custom \
  --mixed-precision
```

## 6. 单张或批量推理

用自己训练好的模型推理：

```bash
python stylize.py \
  --model checkpoints/mosaic_custom_final.pth \
  --input data/content_examples \
  --output-dir outputs/mosaic_custom
```

用下载的公开预训练模型推理：

```bash
python stylize.py \
  --model external_models/mosaic.pth \
  --input data/content_examples \
  --output-dir outputs/mosaic_pretrained
```

## 7. 做公开模型 vs 自训练模型对比

```bash
python compare_models.py \
  --content-dir data/content_examples \
  --baseline-model external_models/mosaic.pth \
  --custom-model checkpoints/mosaic_custom_final.pth \
  --baseline-label pretrained_mosaic \
  --custom-label my_mosaic \
  --output-dir outputs/compare_mosaic
```

输出会生成三联图：

- 左：原图
- 中：公开预训练模型结果
- 右：自己的模型结果


