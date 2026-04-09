# baseline1 说明文档

## 1. 项目简介

`baseline1` 是一套基于 `mmagent` 的轻量级检索问答基线。

它的核心思路是：

1. 从 `VideoGraph` 类型的 `.pkl` 记忆图中抽取文本节点
2. 为文本节点生成外部 embedding
3. 用问题向量对文本节点做相似度检索
4. 按 clip 聚合分数
5. 将检索上下文交给大模型生成答案
6. 再用裁判模型对答案进行自动评测

当前这套 `baseline1` 已经是一个 **character-aware** 版本，也就是：

- 会利用 `pkl` 里的 `character_mappings` / `reverse_character_mappings`
- 会把 `<face_x>` / `<voice_y>` 尽量归一化成 `<character_k>`
- 检索和回答时会显式携带角色归一化信息，提升人物相关问题的鲁棒性

## 2. 目录结构

`baseline1` 目录当前主要包含：

```text
baseline1/
├── README.md
├── data/
│   └── bedroom_01.pkl
├── outputs/
│   ├── bedroom_01_docs.json
│   ├── bedroom_01_vectors.npz
│   ├── bedroom_01_vectors_meta.json
│   ├── bedroom_01_baseline1_results.json
│   ├── bedroom_01_baseline1_results_judged.json
│   └── bedroom_01_baseline1_results_judge_summary.json
└── scripts/
    ├── build_clip_docs.py
    ├── build_clip_embeddings.py
    ├── retrieval.py
    ├── qa_once.py
    ├── run_baseline1.py
    ├── judge_answers.py
    └── inspect_graph.py
```

## 3. 主流程总览

`baseline1` 的完整流程如下：

```text
VideoGraph pkl
   ↓
build_clip_docs.py
   ↓
docs.json
   ↓
build_clip_embeddings.py
   ↓
vectors.npz
   ↓
qa_once.py / run_baseline1.py
   ↓
baseline1_results.json
   ↓
judge_answers.py
   ↓
judged.json + judge_summary.json
```

## 4. 数据流说明

### 4.1 第一步：从 pkl 抽取 docs

脚本：

- [scripts/build_clip_docs.py](/home/ok/m3-agent/baseline1/scripts/build_clip_docs.py)

作用：

- 读取 `VideoGraph` pkl
- 遍历 `text_nodes_by_clip`
- 提取每个文本节点的：
  - `node_id`
  - `type`
  - `timestamp`
  - `content_text`
- 额外导出角色层信息：
  - `character_mappings`
  - `reverse_character_mappings`
- 为每个文本节点生成：
  - `entity_tags_raw`
  - `character_tags`
  - `normalized_content_text`

其中最关键的新增字段是：

- `normalized_content_text`

例如原始文本：

```text
<voice_24> is a designer.
```

如果 `voice_24 -> character_0`，则归一化文本会变成：

```text
<character_0> is a designer.
```

这一步的输出是：

- `outputs/{pkl_name}_docs.json`

### 4.2 第二步：为文本节点建向量

脚本：

- [scripts/build_clip_embeddings.py](/home/ok/m3-agent/baseline1/scripts/build_clip_embeddings.py)

作用：

- 读取 `docs.json`
- 优先使用 `normalized_content_text` 生成 embedding
- 如果没有归一化文本，则回退到 `content_text`
- 输出固定结构的 `.npz`

输出字段包括：

- `embeddings`
- `node_ids`
- `clip_ids`
- `type_ids`

输出文件：

- `outputs/{pkl_name}_vectors.npz`
- `outputs/{pkl_name}_vectors_meta.json`

### 4.3 第三步：执行检索

脚本：

- [scripts/retrieval.py](/home/ok/m3-agent/baseline1/scripts/retrieval.py)

作用：

- 读取 `docs.json` 和 `vectors.npz`
- 根据字段筛选候选节点：
  - `episodic`
  - `semantic`
  - `all`
- 计算 query embedding 与 node embedding 的 cosine similarity
- 按 clip 聚合

当前 clip 打分逻辑保持为：

```text
clip_score = 0.7 * top1 + 0.3 * top2
```

其中：

- `top1` 是该 clip 内相似度最高的 node 分数
- `top2` 是该 clip 内相似度第二高的 node 分数

当前 `retrieval.py` 还会额外输出：

- `normalized_content_text`
- `character_tags`
- `character_hints`

这些信息主要用于后续回答阶段做人物别名理解。

### 4.4 第四步：执行单题问答

脚本：

- [scripts/qa_once.py](/home/ok/m3-agent/baseline1/scripts/qa_once.py)

作用：

1. 用 `answer_model` 先把问题分类成：
   - `episodic`
   - `semantic`
   - `all`
2. 用 `embedding_model` 生成问题向量
3. 调 `retrieval.py` 得到 top clips / top nodes / context
4. 用 `answer_model` 基于检索上下文回答

当前 `qa_once.py` 已做 character-aware 增强：

- answer prompt 会明确告诉模型：
  - 上下文里可能出现 `<face_x>` / `<voice_x>` / `<character_x>`
  - 如果同时给出 raw 和 normalized，优先参考 normalized
  - 不要因为问题里有 Lily / Emma 之类的人名，但上下文里只出现内部标签，就立刻回答 `uncertain`
- prompt 会要求模型先做：
  - `Relevant evidence / alias interpretation`
  - 再给 `Final answer`

### 4.5 第五步：批量跑一个 pkl 的全部问题

脚本：

- [scripts/run_baseline1.py](/home/ok/m3-agent/baseline1/scripts/run_baseline1.py)

作用：

- 给定 `pkl_name` 或 `pkl_path`
- 自动准备 `docs` 和 `vectors`
- 从 `data/annotations/robot.json` 中找到对应问题
- 对每个问题调用 `qa_once.py`
- 最终写出整包结果

输出文件：

- `outputs/{pkl_name}_baseline1_results.json`

### 4.6 第六步：自动判题

脚本：

- [scripts/judge_answers.py](/home/ok/m3-agent/baseline1/scripts/judge_answers.py)

作用：

- 读取 `baseline1_results.json`
- 对每题的：
  - `question`
  - `gold_answer`
  - `pred_answer`
  调用大模型裁判
- 输出每题判定结果与整体 accuracy

当前设计：

- 主裁判模型默认：`Qwen3-Max`
- 可选复核模型默认：`GLM`
- 支持：
  - 主裁判单独使用
  - 条件触发复核
  - `review_all`
  - 合并策略切换

输出文件：

- `outputs/{stem}_judged.json`
- `outputs/{stem}_judge_summary.json`

注意：

- 当前 judged 文件默认是“精简版”
- 不再保留 `retrieved_clips` / `retrieved_nodes` / `context` 等大字段
- 只保留判题和复盘最关键的信息

### 4.7 附加调试工具

脚本：

- [scripts/inspect_graph.py](/home/ok/m3-agent/baseline1/scripts/inspect_graph.py)

作用：

- 不依赖完整 `mmagent` 运行环境时，尽量稳健地读取 pkl
- 打印 `VideoGraph` 的内部结构
- 适合快速查看：
  - 顶层字段
  - 节点总数
  - clip 分布
  - sample node 的内部结构

## 5. 核心脚本说明

下面按脚本逐个总结。

### 5.1 `build_clip_docs.py`

输入：

- `--pkl_path`
- `--output_dir`

输出：

- `{pkl_name}_docs.json`

主要用途：

- 生成 baseline1 的标准 docs 文件
- 导出角色映射信息
- 为后续 character-aware 检索提供归一化文本

### 5.2 `build_clip_embeddings.py`

输入：

- `--docs_path`
- `--output_dir`
- `--embedding_model`
- `--ark_api_key`
- `--batch_size`

输出：

- `{pkl_name}_vectors.npz`
- `{pkl_name}_vectors_meta.json`

主要用途：

- 为文本节点建立向量索引

### 5.3 `retrieval.py`

输入：

- `--field`
- `--query_embedding_path`
- `--docs_path`
- `--vectors_path`
- `--top_k_clips`

输出：

- 可打印 JSON
- 或通过 `--save_path` 保存 retrieval 结果

主要用途：

- 独立调试检索效果

### 5.4 `qa_once.py`

输入：

- `--question`
- `--docs_path`
- `--vectors_path`
- `--answer_model`
- `--embedding_model`
- `--answer_api_key`
- `--answer_base_url`
- `--ark_api_key`

主要用途：

- 调试单道题的完整问答流程

### 5.5 `run_baseline1.py`

输入：

- `--pkl_name` 或 `--pkl_path`
- `--memory_graph_dir`
- `--annotations_dir`
- `--output_dir`
- `--answer_model`
- `--embedding_model`
- `--answer_api_key`
- `--answer_base_url`
- `--ark_api_key`

主要用途：

- 跑一个 pkl 对应的全部问题

### 5.6 `judge_answers.py`

输入：

- `--input_path`
- `--output_dir`
- `--main_model`
- `--main_base_url`
- `--main_api_key`
- `--review_model`
- `--review_base_url`
- `--review_api_key`
- `--enable_review`
- `--review_all`
- `--merge_strategy`

主要用途：

- 自动裁判 baseline1 输出答案

### 5.7 `inspect_graph.py`

输入：

- `--pkl`

主要用途：

- 直接看 pkl 结构

## 6. 环境依赖

这套 baseline1 主要依赖以下几类能力：

- Python 3.9
- `numpy`
- `openai`
- 火山方舟 Ark SDK
- 可访问的 OpenAI-compatible 大模型接口

如果你要完整跑通：

1. 问答模型需要：
   - `answer_api_key`
   - `answer_base_url`
2. embedding 模型需要：
   - `ARK_API_KEY`
3. 自动裁判需要：
   - `QWEN_API_KEY`
   - `QWEN_BASE_URL`
   - 可选 `GLM_API_KEY`
   - 可选 `GLM_BASE_URL`

## 7. 快速开始

### 7.1 查看一个 pkl 的结构

```bash
cd /home/ok/m3-agent

python baseline1/scripts/inspect_graph.py \
  --pkl baseline1/data/bedroom_01.pkl
```

### 7.2 从 pkl 构建 docs

```bash
python baseline1/scripts/build_clip_docs.py \
  --pkl_path baseline1/data/bedroom_01.pkl \
  --output_dir baseline1/outputs
```

### 7.3 从 docs 构建向量

```bash
export ARK_API_KEY="你的_ark_api_key"

python baseline1/scripts/build_clip_embeddings.py \
  --docs_path baseline1/outputs/bedroom_01_docs.json \
  --output_dir baseline1/outputs \
  --embedding_model "你的_embedding_model" \
  --ark_api_key "$ARK_API_KEY"
```

### 7.4 单题调试

```bash
export ANSWER_API_KEY="你的_answer_api_key"
export ANSWER_BASE_URL="你的_answer_base_url"
export ARK_API_KEY="你的_ark_api_key"

python baseline1/scripts/qa_once.py \
  --question "What is Lily's occupation?" \
  --docs_path baseline1/outputs/bedroom_01_docs.json \
  --vectors_path baseline1/outputs/bedroom_01_vectors.npz \
  --answer_model "你的_answer_model" \
  --embedding_model "你的_embedding_model" \
  --answer_api_key "$ANSWER_API_KEY" \
  --answer_base_url "$ANSWER_BASE_URL" \
  --ark_api_key "$ARK_API_KEY"
```

### 7.5 跑一个 pkl 的全部问题

```bash
export ANSWER_API_KEY="你的_answer_api_key"
export ANSWER_BASE_URL="你的_answer_base_url"
export ARK_API_KEY="你的_ark_api_key"

python baseline1/scripts/run_baseline1.py \
  --pkl_path baseline1/data/bedroom_01.pkl \
  --answer_model "你的_answer_model" \
  --embedding_model "你的_embedding_model" \
  --answer_api_key "$ANSWER_API_KEY" \
  --answer_base_url "$ANSWER_BASE_URL" \
  --ark_api_key "$ARK_API_KEY" \
  --output_dir baseline1/outputs
```

### 7.6 自动裁判答案

只用主裁判：

```bash
export QWEN_API_KEY="你的_qwen_api_key"
export QWEN_BASE_URL="你的_qwen_base_url"

python baseline1/scripts/judge_answers.py \
  --input_path baseline1/outputs/bedroom_01_baseline1_results.json \
  --output_dir baseline1/outputs \
  --main_model qwen3-max \
  --main_api_key "$QWEN_API_KEY" \
  --main_base_url "$QWEN_BASE_URL" \
  --enable_review false
```

主裁判 + GLM 复核：

```bash
export QWEN_API_KEY="你的_qwen_api_key"
export QWEN_BASE_URL="你的_qwen_base_url"
export GLM_API_KEY="你的_glm_api_key"
export GLM_BASE_URL="你的_glm_base_url"

python baseline1/scripts/judge_answers.py \
  --input_path baseline1/outputs/bedroom_01_baseline1_results.json \
  --output_dir baseline1/outputs \
  --main_model qwen3-max \
  --main_api_key "$QWEN_API_KEY" \
  --main_base_url "$QWEN_BASE_URL" \
  --review_model glm-4.5 \
  --review_api_key "$GLM_API_KEY" \
  --review_base_url "$GLM_BASE_URL" \
  --enable_review true \
  --merge_strategy review_first
```

## 8. 输出文件说明

### 8.1 `*_docs.json`

主要包含：

- `pkl_name`
- `character_info`
- `clips`

其中每个 node 会保留：

- `node_id`
- `type`
- `timestamp`
- `content_text`
- `entity_tags_raw`
- `character_tags`
- `normalized_content_text`

### 8.2 `*_vectors.npz`

主要包含：

- `embeddings`
- `node_ids`
- `clip_ids`
- `type_ids`

### 8.3 `*_baseline1_results.json`

主要包含：

- 每道题的问题
- gold answer
- pred answer
- field
- retrieved_clips
- retrieved_nodes
- context
- character_hints

这是完整的问答结果包。

### 8.4 `*_judged.json`

主要包含：

- 逐题判定结果
- `correct / incorrect`
- `reason`
- 主裁判和复核裁判原始输出
- 精简版复盘字段

### 8.5 `*_judge_summary.json`

主要包含：

- `total`
- `num_correct`
- `num_incorrect`
- `accuracy`
- `main_judge_model`
- `review_model`
- `review_enabled`

## 9. 设计特点

这套 baseline1 的主要特点是：

- 保持了简单清晰的 node-level retrieval 主线
- 不引入复杂图推理
- 充分利用了 `mmagent` 现成的角色统一层
- 通过 `normalized_content_text` 提升人物相关检索稳定性
- 回答阶段通过 `character_hints` 和 alias interpretation 增强角色理解
- 自动裁判脚本可单独复用，便于后续对比不同 baseline

## 10. 当前限制

当前 baseline1 仍然是一个轻量级 baseline，不是完整图推理系统。

它目前的限制主要有：

- 检索仍主要依赖文本 embedding
- 没有做多跳图推理
- 没有显式建立 `name -> character` 的结构化表
- 对人物名字的理解，仍然依赖文本记忆和大模型推断
- 裁判 accuracy 取决于裁判模型本身，不等价于人工标注

## 11. 建议阅读顺序

如果你想快速理解这套 baseline1，推荐按下面顺序阅读：

1. [scripts/run_baseline1.py](/home/ok/m3-agent/baseline1/scripts/run_baseline1.py)
2. [scripts/qa_once.py](/home/ok/m3-agent/baseline1/scripts/qa_once.py)
3. [scripts/retrieval.py](/home/ok/m3-agent/baseline1/scripts/retrieval.py)
4. [scripts/build_clip_docs.py](/home/ok/m3-agent/baseline1/scripts/build_clip_docs.py)
5. [scripts/build_clip_embeddings.py](/home/ok/m3-agent/baseline1/scripts/build_clip_embeddings.py)
6. [scripts/judge_answers.py](/home/ok/m3-agent/baseline1/scripts/judge_answers.py)

如果你想先看 pkl 结构，再看 baseline1，推荐先读：

1. [scripts/inspect_graph.py](/home/ok/m3-agent/baseline1/scripts/inspect_graph.py)
2. [MMAGENT_PKL_STRUCTURE_CN.md](/home/ok/m3-agent/MMAGENT_PKL_STRUCTURE_CN.md)

## 12. 一句话总结

`baseline1` 是一套面向 `mmagent` 视频记忆图的、轻量但实用的 **character-aware 检索问答 baseline**：

- 用 `docs + embeddings + retrieval + answer_model` 跑问答
- 用 `judge_answers.py` 做自动评测
- 尽量在不推翻原主线的前提下，把角色归一化信息利用起来

