# Image Style Transfer Coursework

本项目为“风格迁移实验”作业准备了一个可直接使用的 PyTorch 工程，目标是：

1. 输入自己的照片或喜欢的照片。
2. 至少实现两种艺术风格迁移。
3. 训练自己的风格迁移模型。
4. 和网上下载的已训练模型做可复现实验对比。

## 1. 方法选择

项目采用 `Johnson et al., Perceptual Losses for Real-Time Style Transfer` 的快速风格迁移方案：

- 优点：训练完成后推理很快，适合课程作业展示。
- 缺点：一个模型通常只对应一种风格，所以如果要两种风格，通常训练两个模型。
- 公平对比方式：下载相同架构的公开 `.pth` 权重，用同一批内容图和尽量相同的数据集设置比较。

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

推荐使用：

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
  - 仓库提供 MS-COCO 训练说明与预训练模型下载脚本，适合做课程对比实验。

建议你优先下载以下风格的 `.pth` 权重到 `external_models/`：

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

这样非常适合直接放进实验报告。

## 8. 单卡 5080 16GB 的建议

由于你训练发生在 Windows 11 的 WSL2 中，建议先用下面的保守配置：

- `--image-size 256`
- `--style-size 512`
- `--batch-size 8`
- `--mixed-precision`
- `--subset-size 20000`
- `--num-workers 4`

如果显存不够，按顺序调整：

1. 先把 `batch-size` 从 `8` 改到 `4`
2. 再把 `image-size` 从 `256` 改到 `224`
3. 再减少 `num-workers`

如果还有余量，按顺序提高效果：

1. 把 `subset-size` 从 `20000` 提到 `30000` 或 `40000`
2. 把 `epochs` 从 `2` 提到 `3`
3. 视情况把 `style-weight` 从 `1e5` 提到 `2e5`

## 9. 关键超参数怎么调

代码中已经在 `train.py` 的参数定义位置写了注释。这里再总结一次：

- `style-weight`
  - 调大：风格更强、颜色和纹理更明显
  - 调小：更保留原图结构
- `content-weight`
  - 调大：原图轮廓和物体结构更稳定
  - 调小：更容易被风格覆盖
- `tv-weight`
  - 调大：图像更平滑，噪点和棋盘格更少
  - 调太大：画面会发糊
- `image-size`
  - 调大：细节更好，但更吃显存和时间
  - 调小：训练更稳更快
- `subset-size`
  - 调大：泛化更好
  - 调小：更快出结果，适合先做课程实验
- `learning-rate`
  - 太大：loss 抖动，结果不稳定
  - 太小：收敛慢

## 10. 建议实验设计

建议至少做下面两组实验：

1. `mosaic` 风格：
   - 公开模型：`external_models/mosaic.pth`
   - 自训练模型：`checkpoints/mosaic_custom_final.pth`
2. `candy` 风格：
   - 公开模型：`external_models/candy.pth`
   - 自训练模型：`checkpoints/candy_custom_final.pth`

这样就满足“至少两种艺术风格”的要求。

## 11. 报告撰写建议

报告可以按 `docs/report_template.md` 直接填写，建议配图包括：

- 原图
- 两种风格迁移结果
- 公开模型 vs 自训练模型对比图
- 训练中间 preview 图

## 12. 你需要自己下载的内容

在本机上我没有替你下载数据和权重，但你只需要准备：

- `MS-COCO` 训练集到 `data/coco_train2014/`
- 两张风格图到 `data/style_images/`
- 至少一个公开预训练 `.pth` 到 `external_models/`

准备完后就可以直接训练和对比。
