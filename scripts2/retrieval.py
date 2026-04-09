#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个文件的主要功能：
读取 docs.json 与 vectors.npz，基于 query embedding 做节点级检索，
再按 clip 聚合，选出 top-3 clip，并构造 answer_model 可直接使用的 context。

说明：
- 检索统一使用外部 embedding_model 生成的向量。
- 不使用 pkl 中原始 node.embeddings。
- 支持 import 调用，也支持命令行单独运行。
- retrieval 现在是 character-aware 的：打分主逻辑不变，但输出会包含角色归一化信息，方便 answer_model 推理。
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("retrieval")
TYPE_TO_ID = {"episodic": 0, "semantic": 1}
ID_TO_TYPE = {0: "episodic", 1: "semantic"}
VALID_FIELDS = {"episodic", "semantic", "all"}
CHARACTER_TAG_PATTERN = re.compile(r"<character_\d+>")


def setup_logging() -> None:
    """初始化日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """按首次出现顺序去重。"""
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def load_docs_and_vectors(docs_path: str, vectors_path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    同时读取 docs.json 与 vectors.npz。

    输入：
    - docs_path: docs json 路径
    - vectors_path: vectors npz 路径

    输出：
    - docs 数据字典
    - vectors 数据字典
    """
    docs_file = Path(docs_path)
    vectors_file = Path(vectors_path)

    if not docs_file.is_file():
        raise FileNotFoundError("Docs json not found: {0}".format(docs_file))
    if not vectors_file.is_file():
        raise FileNotFoundError("Vectors npz not found: {0}".format(vectors_file))

    LOGGER.info("Loading docs from %s", docs_file)
    with docs_file.open("r", encoding="utf-8") as file_obj:
        docs_data = json.load(file_obj)

    LOGGER.info("Loading vectors from %s", vectors_file)
    npz_data = np.load(vectors_file)
    vectors_data = {
        "embeddings": np.asarray(npz_data["embeddings"], dtype=np.float32),
        "node_ids": np.asarray(npz_data["node_ids"]),
        "clip_ids": np.asarray(npz_data["clip_ids"]),
        "type_ids": np.asarray(npz_data["type_ids"]),
    }
    return docs_data, vectors_data


def get_character_info(docs_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    从 docs 顶层读取角色映射信息。
    """
    raw_character_info = docs_data.get("character_info", {}) or {}
    raw_character_mappings = raw_character_info.get("character_mappings", {}) or {}
    raw_reverse_character_mappings = raw_character_info.get("reverse_character_mappings", {}) or {}

    character_mappings: Dict[str, List[str]] = {}
    for key, value in raw_character_mappings.items():
        aliases = value if isinstance(value, (list, tuple)) else [value]
        character_mappings[str(key)] = [str(alias) for alias in aliases]

    reverse_character_mappings = {str(key): str(value) for key, value in raw_reverse_character_mappings.items()}
    return {
        "character_mappings": character_mappings,
        "reverse_character_mappings": reverse_character_mappings,
    }


def build_node_lookup(docs_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    基于 docs.json 建立 node_id -> node_info 的索引。
    """
    clips = docs_data.get("clips", [])
    lookup: Dict[int, Dict[str, Any]] = {}
    for clip_item in clips:
        clip_id = int(clip_item["clip_id"])
        for node_item in clip_item.get("nodes", []):
            node_id = int(node_item["node_id"])
            lookup[node_id] = {
                "clip_id": clip_id,
                "node_id": node_id,
                "type": str(node_item["type"]),
                "timestamp": int(node_item["timestamp"]),
                "content_text": str(node_item.get("content_text", "")),
                "normalized_content_text": str(node_item.get("normalized_content_text", "")),
                "entity_tags_raw": list(node_item.get("entity_tags_raw", [])),
                "character_tags": list(node_item.get("character_tags", [])),
            }
    return lookup


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    计算 query 与矩阵每一行的 cosine similarity。

    输入：
    - query: shape=(dim,)
    - matrix: shape=(num_nodes, dim)

    输出：
    - shape=(num_nodes,) 的相似度数组
    """
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D, got shape={0}".format(matrix.shape))
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Query dim does not match embedding dim. query_dim={0}, embedding_dim={1}".format(
                query.shape[0], matrix.shape[1]
            )
        )

    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(query_norm * matrix_norm, 1e-12)
    scores = np.dot(matrix, query) / denominator
    return scores.astype(np.float32)


def filter_indices_by_field(type_ids: np.ndarray, field: str) -> np.ndarray:
    """
    按 field 过滤候选节点索引。

    输入：
    - type_ids: 所有节点的 type_id 数组
    - field: episodic / semantic / all

    输出：
    - 候选索引数组
    """
    if field not in VALID_FIELDS:
        raise ValueError("Unsupported field: {0}".format(field))

    if field == "all":
        return np.arange(len(type_ids))
    if field == "episodic":
        return np.where(type_ids == TYPE_TO_ID["episodic"])[0]
    return np.where(type_ids == TYPE_TO_ID["semantic"])[0]


def group_node_scores_by_clip(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    node_ids: np.ndarray,
    clip_ids: np.ndarray,
    type_ids: np.ndarray,
    node_lookup: Dict[int, Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    将候选节点分数按 clip 分组。
    """
    grouped: Dict[int, List[Dict[str, Any]]] = {}

    for index in candidate_indices:
        node_id = int(node_ids[index])
        clip_id = int(clip_ids[index])
        node_info = dict(node_lookup.get(node_id, {}))
        if not node_info:
            LOGGER.warning("node_id=%s not found in docs lookup, skip.", node_id)
            continue

        node_info["score"] = float(scores[index])
        node_info["type"] = ID_TO_TYPE.get(int(type_ids[index]), node_info.get("type", "unknown"))
        node_info["entity_tags_raw"] = list(node_info.get("entity_tags_raw", []))
        node_info["character_tags"] = list(node_info.get("character_tags", []))
        grouped.setdefault(clip_id, []).append(node_info)

    for clip_id, items in grouped.items():
        items.sort(key=lambda item: (-item["score"], item["timestamp"], item["node_id"]))
        LOGGER.info("clip_id=%s has %s candidate nodes after filtering.", clip_id, len(items))

    return grouped


def build_retrieved_clips(grouped_scores: Dict[int, List[Dict[str, Any]]], top_k_clips: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从按 clip 分组的候选节点中构造最终检索结果。

    规则：
    - 每个 clip 取最高的两个 node
    - clip_score = 0.7 * top1 + 0.3 * top2
    - 若 clip 少于两个 node，则使用 top1
    """
    clip_items: List[Dict[str, Any]] = []
    flat_nodes: List[Dict[str, Any]] = []

    for clip_id, node_items in grouped_scores.items():
        if not node_items:
            continue
        top_nodes = [dict(node_item) for node_item in node_items[:2]]
        if len(top_nodes) >= 2:
            clip_score = float(0.7 * top_nodes[0]["score"] + 0.3 * top_nodes[1]["score"])
        else:
            clip_score = float(top_nodes[0]["score"])
        clip_record = {
            "clip_id": int(clip_id),
            "clip_score": clip_score,
            "top_nodes": top_nodes,
        }
        clip_items.append(clip_record)

    clip_items.sort(key=lambda item: (-item["clip_score"], item["clip_id"]))
    retrieved_clips = clip_items[: max(int(top_k_clips), 1)]

    for clip_item in retrieved_clips:
        for node_item in clip_item["top_nodes"]:
            flat_node = dict(node_item)
            flat_node["clip_id"] = int(clip_item["clip_id"])
            flat_node["clip_score"] = float(clip_item["clip_score"])
            flat_nodes.append(flat_node)

    return retrieved_clips, flat_nodes


def format_character_tags(character_tags: List[str]) -> str:
    """将 character_tags 列表格式化为展示字符串。"""
    if not character_tags:
        return ""
    return ", ".join(character_tags)


def build_context_string(retrieved_clips: List[Dict[str, Any]]) -> str:
    """
    将 retrieved_clips 整理成适合 answer_model 直接使用的上下文字符串。

    说明：
    - 若 normalized_content_text 与 content_text 相同，则不重复展示
    - 若不同，则同时展示 raw 和 normalized
    """
    blocks: List[str] = []
    for clip_item in retrieved_clips:
        lines = ["Clip {0}, score={1:.4f}".format(clip_item["clip_id"], clip_item["clip_score"])]
        for node_item in clip_item.get("top_nodes", []):
            lines.append(
                "- Node {0} [{1}, t={2}, score={3:.4f}]".format(
                    node_item["node_id"],
                    node_item["type"],
                    node_item["timestamp"],
                    node_item["score"],
                )
            )

            raw_text = str(node_item.get("content_text", "")).strip()
            normalized_text = str(node_item.get("normalized_content_text", "")).strip()
            if normalized_text and normalized_text != raw_text:
                lines.append("  raw: {0}".format(raw_text))
                lines.append("  normalized: {0}".format(normalized_text))
            else:
                lines.append("  text: {0}".format(normalized_text or raw_text))

            character_tags_text = format_character_tags(list(node_item.get("character_tags", [])))
            if character_tags_text:
                lines.append("  character_tags: {0}".format(character_tags_text))

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()


def build_character_hints(
    retrieved_nodes: List[Dict[str, Any]],
    character_mappings: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    仅基于当前命中的 top_nodes 汇总 character 信息。

    输出示例：
    [
      {
        "character_tag": "<character_0>",
        "aliases": ["face_3", "voice_24"]
      }
    ]
    """
    hint_items: List[Dict[str, Any]] = []
    seen_character_keys = set()

    for node_item in retrieved_nodes:
        for character_tag in node_item.get("character_tags", []):
            if not CHARACTER_TAG_PATTERN.fullmatch(str(character_tag).strip()):
                continue
            character_key = str(character_tag).strip()[1:-1]
            if character_key in seen_character_keys:
                continue
            seen_character_keys.add(character_key)
            aliases = dedupe_preserve_order(character_mappings.get(character_key, []))
            hint_items.append(
                {
                    "character_tag": "<{0}>".format(character_key),
                    "aliases": aliases,
                }
            )

    return hint_items


def retrieve(
    field: str,
    query_embedding: np.ndarray,
    docs_path: str,
    vectors_path: str,
    top_k_clips: int = 3,
) -> Dict[str, Any]:
    """
    baseline1 检索主流程。

    输入：
    - field: episodic / semantic / all
    - query_embedding: 查询向量
    - docs_path: docs json 路径
    - vectors_path: vectors npz 路径
    - top_k_clips: 返回 top-k clip

    输出：
    - 包含 retrieved_clips / retrieved_nodes / context / character_hints 的字典
    """
    docs_data, vectors_data = load_docs_and_vectors(docs_path, vectors_path)
    node_lookup = build_node_lookup(docs_data)
    character_info = get_character_info(docs_data)

    embeddings = vectors_data["embeddings"]
    node_ids = vectors_data["node_ids"]
    clip_ids = vectors_data["clip_ids"]
    type_ids = vectors_data["type_ids"]

    if not (len(embeddings) == len(node_ids) == len(clip_ids) == len(type_ids)):
        raise ValueError("vectors arrays length mismatch.")

    candidate_indices = filter_indices_by_field(type_ids, field)
    LOGGER.info("Field=%s, candidate node count=%s", field, len(candidate_indices))
    if len(candidate_indices) == 0:
        return {
            "retrieved_clips": [],
            "retrieved_nodes": [],
            "context": "",
            "character_hints": [],
        }

    scores = cosine_similarity(query_embedding, embeddings)
    grouped_scores = group_node_scores_by_clip(
        candidate_indices=candidate_indices,
        scores=scores,
        node_ids=node_ids,
        clip_ids=clip_ids,
        type_ids=type_ids,
        node_lookup=node_lookup,
    )
    retrieved_clips, retrieved_nodes = build_retrieved_clips(grouped_scores, top_k_clips=top_k_clips)
    context = build_context_string(retrieved_clips)
    character_hints = build_character_hints(
        retrieved_nodes=retrieved_nodes,
        character_mappings=character_info["character_mappings"],
    )

    LOGGER.info("Retrieved top-%s clips.", len(retrieved_clips))
    for clip_item in retrieved_clips:
        LOGGER.info("Top clip_id=%s score=%.4f", clip_item["clip_id"], clip_item["clip_score"])

    return {
        "retrieved_clips": retrieved_clips,
        "retrieved_nodes": retrieved_nodes,
        "context": context,
        "character_hints": character_hints,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Retrieve top clips from docs/vectors with a query embedding.")
    parser.add_argument("--field", default="all", choices=["episodic", "semantic", "all"], help="Retrieval field.")
    parser.add_argument("--query_embedding_path", required=True, help="Path to a .npy file storing one query embedding.")
    parser.add_argument("--docs_path", required=True, help="Path to docs json.")
    parser.add_argument("--vectors_path", required=True, help="Path to vectors npz.")
    parser.add_argument("--top_k_clips", type=int, default=3, help="Number of top clips to return.")
    parser.add_argument("--save_path", default="", help="Optional path to save retrieval result json.")
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""
    setup_logging()
    args = parse_args()

    try:
        query_embedding = np.load(args.query_embedding_path)
        result = retrieve(
            field=args.field,
            query_embedding=query_embedding,
            docs_path=args.docs_path,
            vectors_path=args.vectors_path,
            top_k_clips=args.top_k_clips,
        )

        if args.save_path:
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as file_obj:
                json.dump(result, file_obj, ensure_ascii=False, indent=2)
            LOGGER.info("Saved retrieval result to %s", save_path)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        LOGGER.exception("Retrieval failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
