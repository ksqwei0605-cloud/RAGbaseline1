#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个文件的主要功能：
从 VideoGraph pkl 中提取 baseline1 需要的文本节点，并整理成统一的 docs.json。

说明：
- 只从 graph.text_nodes_by_clip 出发提取文本节点。
- 原始 pkl 中的文本来自 node.metadata["contents"]，但写入 docs.json 时统一整理为 content_text。
- 这个文件现在除了抽取 clip docs，还负责导出角色映射信息，并为每个文本 node 生成角色归一化文本。
- 不使用 pkl 里原始 node.embeddings 做检索；这里只负责整理文本与元信息。
"""

import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("build_clip_docs")
VALID_TEXT_NODE_TYPES = {"episodic", "semantic"}
ENTITY_TAG_PATTERN = re.compile(r"<(?:face|voice|character)_\d+>")


def setup_logging() -> None:
    """初始化日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


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


def load_video_graph(pkl_path: str) -> Any:
    """
    读取 VideoGraph pkl 文件。

    输入：
    - pkl_path: pkl 文件路径

    输出：
    - 反序列化后的 VideoGraph 对象
    """
    path = Path(pkl_path)
    if not path.is_file():
        raise FileNotFoundError("PKL file not found: {0}".format(path))

    LOGGER.info("Loading VideoGraph from %s", path)
    with path.open("rb") as file_obj:
        graph = pickle.load(file_obj)
    return graph


def normalize_content(contents: Any) -> str:
    """
    将 node.metadata["contents"] 整理为统一字符串 content_text。

    输入：
    - contents: 原始文本内容，通常为 list[str]

    输出：
    - 清洗后的字符串；若无有效文本则返回空字符串
    """
    if contents is None:
        return ""

    if isinstance(contents, str):
        return contents.strip()

    if isinstance(contents, (list, tuple)):
        cleaned_parts: List[str] = []
        for item in contents:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                cleaned_parts.append(text)
        return " ".join(cleaned_parts).strip()

    return str(contents).strip()


def extract_entity_tags(content_text: str) -> List[str]:
    """
    从文本中提取实体标签。

    支持：
    - <face_x>
    - <voice_x>
    - <character_x>
    """
    if not content_text:
        return []
    return dedupe_preserve_order(ENTITY_TAG_PATTERN.findall(content_text))


def normalize_entity_tag_to_character(tag: str, reverse_character_mappings: Dict[str, str]) -> str:
    """
    将单个实体标签归一化为角色标签。

    规则：
    - <character_x> 保持不变
    - <face_x> / <voice_x> 如果能在 reverse_character_mappings 中找到映射，则转成 <character_k>
    - 否则保持原样
    """
    tag = str(tag).strip()
    if not tag:
        return tag

    if re.fullmatch(r"<character_\d+>", tag):
        return tag

    if not (tag.startswith("<") and tag.endswith(">")):
        return tag

    entity_key = tag[1:-1]
    character_key = str(reverse_character_mappings.get(entity_key, "")).strip()
    if not character_key:
        return tag
    return "<{0}>".format(character_key)


def build_normalized_content_text(content_text: str, reverse_character_mappings: Dict[str, str]) -> str:
    """
    将文本中的 <face_x> / <voice_x> 尽可能替换为 <character_k>。

    说明：
    - 这是一个独立函数，便于后续继续调整替换策略。
    - 如果某个标签找不到 reverse mapping，则保持原样。
    """
    if not content_text:
        return ""

    def replace_tag(match: re.Match[str]) -> str:
        raw_tag = match.group(0)
        return normalize_entity_tag_to_character(raw_tag, reverse_character_mappings)

    return ENTITY_TAG_PATTERN.sub(replace_tag, content_text)


def get_node_timestamp(node: Any) -> int:
    """提取节点时间戳。"""
    metadata = getattr(node, "metadata", {}) or {}
    timestamp = metadata.get("timestamp")
    if timestamp is None:
        raise ValueError("Node {0} does not contain timestamp in metadata.".format(getattr(node, "id", "unknown")))
    return int(timestamp)


def sanitize_character_info(graph: Any) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    提取并清洗 graph 中的 character 映射信息。

    返回：
    - character_mappings
    - reverse_character_mappings
    """
    raw_character_mappings = getattr(graph, "character_mappings", {}) or {}
    raw_reverse_character_mappings = getattr(graph, "reverse_character_mappings", {}) or {}

    character_mappings: Dict[str, List[str]] = {}
    for key, value in raw_character_mappings.items():
        aliases = value if isinstance(value, (list, tuple)) else [value]
        cleaned_aliases = dedupe_preserve_order(str(alias).strip() for alias in aliases if str(alias).strip())
        character_mappings[str(key).strip()] = cleaned_aliases

    reverse_character_mappings: Dict[str, str] = {}
    for key, value in raw_reverse_character_mappings.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            reverse_character_mappings[key_text] = value_text

    # 如果 reverse_character_mappings 缺失，尽量从 character_mappings 补齐，保持下游稳健。
    if character_mappings and not reverse_character_mappings:
        for character_key, aliases in character_mappings.items():
            for alias in aliases:
                reverse_character_mappings[str(alias).strip()] = str(character_key).strip()

    return character_mappings, reverse_character_mappings


def build_node_record(node: Any, reverse_character_mappings: Dict[str, str]) -> Dict[str, Any]:
    """
    将单个文本节点整理成 docs.json 中的 node 结构。

    额外新增字段：
    - entity_tags_raw
    - character_tags
    - normalized_content_text
    """
    node_type = str(getattr(node, "type", "")).strip()
    if node_type not in VALID_TEXT_NODE_TYPES:
        raise ValueError("Unsupported text node type: {0}".format(node_type))

    metadata = getattr(node, "metadata", {}) or {}
    content_text = normalize_content(metadata.get("contents"))
    if not content_text:
        return {}

    entity_tags_raw = extract_entity_tags(content_text)
    character_tags = dedupe_preserve_order(
        normalize_entity_tag_to_character(tag, reverse_character_mappings) for tag in entity_tags_raw
    )
    normalized_content_text = build_normalized_content_text(content_text, reverse_character_mappings)

    return {
        "node_id": int(getattr(node, "id")),
        "type": node_type,
        "timestamp": get_node_timestamp(node),
        "content_text": content_text,
        "entity_tags_raw": entity_tags_raw,
        "character_tags": character_tags,
        "normalized_content_text": normalized_content_text,
    }


def extract_clip_docs(graph: Any, pkl_name: str) -> Dict[str, Any]:
    """
    从 VideoGraph 中提取按 clip 分组的文本节点文档。

    输出结构：
    - pkl_name
    - character_info
    - clips
    """
    nodes = getattr(graph, "nodes", None)
    text_nodes_by_clip = getattr(graph, "text_nodes_by_clip", None)

    if not isinstance(nodes, dict):
        raise ValueError("graph.nodes is missing or not a dict.")
    if not isinstance(text_nodes_by_clip, dict):
        raise ValueError("graph.text_nodes_by_clip is missing or not a dict.")

    character_mappings, reverse_character_mappings = sanitize_character_info(graph)

    clips: List[Dict[str, Any]] = []
    total_kept_nodes = 0
    total_skipped_nodes = 0

    for clip_id in sorted(text_nodes_by_clip.keys()):
        raw_node_ids = text_nodes_by_clip.get(clip_id, [])
        if not isinstance(raw_node_ids, (list, tuple)):
            LOGGER.warning("clip_id=%s has invalid node list, skipping clip.", clip_id)
            continue

        clip_nodes: List[Dict[str, Any]] = []
        for node_id in raw_node_ids:
            node = nodes.get(node_id)
            if node is None:
                LOGGER.warning("Missing node_id=%s in graph.nodes, skip.", node_id)
                total_skipped_nodes += 1
                continue

            node_type = str(getattr(node, "type", "")).strip()
            if node_type not in VALID_TEXT_NODE_TYPES:
                LOGGER.warning("clip_id=%s node_id=%s has unsupported type=%s, skip.", clip_id, node_id, node_type)
                total_skipped_nodes += 1
                continue

            try:
                node_record = build_node_record(node, reverse_character_mappings)
            except Exception as exc:
                LOGGER.warning("Failed to parse clip_id=%s node_id=%s: %s", clip_id, node_id, exc)
                total_skipped_nodes += 1
                continue

            if not node_record:
                LOGGER.info("clip_id=%s node_id=%s content_text is empty, skip.", clip_id, node_id)
                total_skipped_nodes += 1
                continue

            clip_nodes.append(node_record)
            total_kept_nodes += 1

        clip_nodes.sort(key=lambda item: (item["timestamp"], item["node_id"]))
        clips.append(
            {
                "clip_id": int(clip_id),
                "nodes": clip_nodes,
            }
        )
        LOGGER.info("Processed clip_id=%s with %s valid nodes.", clip_id, len(clip_nodes))

    LOGGER.info(
        "Finished extracting docs: %s clips, %s kept nodes, %s skipped nodes.",
        len(clips),
        total_kept_nodes,
        total_skipped_nodes,
    )

    return {
        "pkl_name": pkl_name,
        "character_info": {
            "character_mappings": character_mappings,
            "reverse_character_mappings": reverse_character_mappings,
        },
        "clips": clips,
    }


def save_docs_json(data: Dict[str, Any], out_path: str) -> None:
    """
    保存 docs.json。

    输入：
    - data: docs 数据
    - out_path: 输出文件路径
    """
    path = Path(out_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("Saved docs json to %s", path)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build baseline1 docs json from a VideoGraph pkl file.")
    parser.add_argument("--pkl_path", required=True, help="Path to the input VideoGraph pkl file.")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "baseline1" / "outputs"),
        help="Directory to save outputs/{pkl_name}_docs.json.",
    )
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""
    setup_logging()
    args = parse_args()

    pkl_path = Path(args.pkl_path)
    pkl_name = pkl_path.stem
    output_dir = Path(args.output_dir)
    out_path = output_dir / "{0}_docs.json".format(pkl_name)

    try:
        graph = load_video_graph(str(pkl_path))
        docs_data = extract_clip_docs(graph, pkl_name=pkl_name)
        save_docs_json(docs_data, str(out_path))
    except Exception as exc:
        LOGGER.exception("Failed to build docs json: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
