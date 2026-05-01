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
    ├── content_examples/
    └── style_images/

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

## 8. 项目结果总结
通过这次实验，我对快速风格迁移有了一个比一开始更具体的认识：这类方法真正困难的地方，并不只是“把代码跑通”或者“把损失降下来”，而是在风格表达和内容保留之间找到一个比较合适的平衡点。刚开始我会比较自然地认为，只要继续调 ‘style_weight’、‘content_weight’ 这些超参数，效果应该就能慢慢逼近公开模型；但实际做下来发现，很多问题并不是简单靠调大或调小某个权重就能解决的。例如，当我想让风格更明显时，图像中的纹理和颜色迁移确实会更强，但文字、栏杆、边缘这些高频结构也更容易被破坏；而当我为了保住文字去降低风格强度时，整体图像又会显得“有一点风格，但不够像一幅真正风格化的作品”。这让我意识到，风格迁移任务里最核心的问题不是某一个指标做到最好，而是多个目标之间的权衡。  
另一个比较深的体会是，公开实现之所以效果更好，往往不是因为“它只比我多调了一点超参数”，而是它在网络结构、损失设计、训练经验和工程细节上都更加成熟。在前两轮实验中，我已经能够复现出基本可用的风格迁移结果，但和 ‘pytorch-neural’对比后可以明显看到差距，比如太阳周围的风格纹理不够充分、某些局部风格分布不均、文字和细边缘容易发糊。这说明对于图像风格迁移这种任务，仅仅拥有一个标准 ‘Johnson’框架还不够，还需要结合具体问题去分析模型到底是“风格不够强”，还是“保内容过头”，还是“对局部细节缺乏约束”。也正因为这样，后续实验里我才逐步尝试引入跳连、细节分支、门控融合和边缘保持损失，希望把这些比较具体的问题拆开处理，而不是继续用一组统一超参数去硬拉结果。  
从实验过程本身来看，我觉得这次三轮迭代最有价值的地方在于：它让我认识到深度学习实验并不是一次性得到正确答案，而更像是一个持续定位问题、提出假设、验证修改的过程。第一轮实验让我知道基础结构能做出什么效果，也让我看到它的明显短板；第二轮实验说明，结构改动虽然能改善局部细节，但也可能带来新的副作用，比如风格区域分布不均、某些本应更强烈风格化的位置反而被保守处理；第三轮实验则进一步让我理解，网络结构和损失函数应该配合设计，单纯改其中一个往往不够。比如，如果只靠跳连去保细节，模型可能会偏向保内容；但如果再加入边缘保持损失和分层风格约束，就能更有针对性地保护文字和轮廓，同时让低层纹理风格更充分地表现出来。
总体来说，这次实验收获颇丰，对一个具体问题的深入求知是学习深度学习的最好路径。
