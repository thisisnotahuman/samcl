# SAMCL (Minimal Research Framework)

目标：在**相同模型与相同对比学习目标**下，仅改变“语义感知采样策略”，研究其对**样本效率**（performance vs #samples）的影响。

## 关键约束（不违反）

- **无**跨模态融合、无 cross-attention、无 joint encoder
- teacher 仅用于**采样**：**冻结**、**不反传**、**不进入loss**
- 训练目标固定为 CLIP-style InfoNCE（I→T 与 T→I）

## 目录结构

- `src/samcl/models/`：CLIP-style 双编码器（image/text，预训练且训练时不冻结）
- `src/samcl/data/`：COCO image-caption pairs + (image_id, caption_id) 管理
- `src/samcl/teachers/`：冻结 teacher 编码器（text/image）
- `src/samcl/semantic/`：`get_semantic_relation(image_id, caption_id)` 核心抽象
- `src/samcl/sampling/`：随机 vs 语义采样 BatchSampler（四类关系可配比）
- `src/samcl/eval/`：检索评估 Recall@K
- `src/samcl/train.py`：训练入口（固定预算，记录batch组成）

## 快速开始

1) 安装依赖（建议在 venv/conda 中）：

```bash
pip install -r requirements.txt
pip install -e .
```

2) 准备 COCO 2017 captions 数据（自动下载到 `../data/`）：

COCO captions 的“现成数据”就是官方 COCO 2017 的 **images** + **captions annotations**。本项目需要的就是：

- **图片目录**：`train2017/`（或 `val2017/`）
- **caption 标注 json**：`annotations/captions_train2017.json`（或 `captions_val2017.json`）

可以用脚本一键下载并解压（来源：COCO 官方镜像 `images.cocodataset.org`）：

```bash
python scripts/prepare_coco2017.py --all
```

默认落盘结构：

- `../data/coco2017/images/train2017/`
- `../data/coco2017/annotations/captions_train2017.json`

3) 训练（示例）：

```bash
python -m samcl.train \
  --coco_images_dir ../data/coco2017/images/train2017 \
  --coco_captions_json ../data/coco2017/annotations/captions_train2017.json \
  --vision_model_name google/vit-base-patch16-224-in21k \
  --text_model_name bert-base-uncased \
  --proj_dim 256 --proj_hidden_dim 512 --proj_layers 2 \
  --seed 0 \
  --batch_size 128 \
  --max_steps 2000 \
  --eval_every 200 --eval_max_pairs 5000 \
  --sampling_strategy semantic \
  --mix_r1 0.10 --mix_r2 0.30 --mix_r3 0.30 --mix_r4 0.30 \
  --text_sim_threshold 0.55 --image_sim_threshold 0.70 \
  --cache_dir ./cache
```

4) Baseline（随机采样）：

```bash
python -m samcl.train \
  --coco_images_dir ../data/coco2017/images/train2017 \
  --coco_captions_json ../data/coco2017/annotations/captions_train2017.json \
  --sampling_strategy random
```

## 分析：绘制 teacher 相似度分布

teacher embedding 在第一次训练/运行时会缓存到 `--cache_dir`，只要 **teacher 配置与数据不变**，caption/image embedding 就不变；因此相似度分布也不会变（同一采样 seed 下）。

用脚本随机采样大量 caption–caption 与 image–image pair，绘制 cosine similarity 的直方图，并把采样结果缓存到 `cache_dir/similarity_samples/`：

```bash
python scripts/plot_teacher_similarity.py \
  --coco_images_dir ../data/coco2017/images/train2017 \
  --coco_captions_json ../data/coco2017/annotations/captions_train2017.json \
  --cache_dir ./cache \
  --teacher_text_model sentence-transformers/all-MiniLM-L6-v2 \
  --teacher_image_arch clip --teacher_image_model openai/clip-vit-base-patch32 \
  --num_caption_pairs 300000 --num_image_pairs 300000 \
  --out_dir ./plots
```

## 分析：阈值可视化（相似 vs 不相似长什么样）

用 teacher embedding 在给定阈值下，随机选 query image，展示：
- query image + 它的 GT captions
- **相似图像**（cosine ≥ `image_sim_threshold`）与 **不相似图像**（cosine < 阈值）
- 选一个 query caption，展示 **相似 captions / 不相似 captions**（并附带对应图片缩略图）

输出是一个 HTML 报告（带缩略图）：`out_dir/report.html`。

```bash
python scripts/visualize_teacher_thresholds.py \
  --coco_images_dir ../data/coco2017/images/train2017 \
  --coco_captions_json ../data/coco2017/annotations/captions_train2017.json \
  --cache_dir ./cache \
  --teacher_text_model sentence-transformers/all-MiniLM-L6-v2 \
  --teacher_image_arch clip --teacher_image_model openai/clip-vit-base-patch32 \
  --text_sim_threshold 0.30 --image_sim_threshold 0.65 \
  --num_queries 3 \
  --out_dir ./viz_thresholds
```

## 语义关系定义（核心）

对任意 image–text pair \((i, t)\)：

- text “相似/不同”：caption \(t\) 与 image \(i\) 的 caption 集合之间的 teacher text 相似度（max over captions(i)）
- image “相似/不同”：image \(i\) 与 caption \(t\) 所属 image \(j\) 之间的 teacher image 相似度

从而得到四类：

1. 相似文本 + 相似图像
2. 相似文本 + 不同图像
3. 不同文本 + 相似图像
4. 不同文本 + 不同图像（trivial negative）

> 训练 loss 不变；采样器通过控制 batch 中样本共现，改变 batch 内 cross-pairs 的语义组成。

