#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build clip-level text documents with embeddings from VideoGraph pickle.

Usage:
  python build_clip_docs.py --pkl path/to/memory_graph.pkl --out clip_docs_with_embeddings.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


TEXT_FIELD_CANDIDATES = ["content", "text", "value", "raw_content", "data", "contents"]

# ========== 路径初始化 ==========
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)


# ========== 短标签集合 ==========
LABEL_KEYWORDS = {
    "episodic", "semantic", "text", "voice", "face", "img", "image",
    "audio", "memory", "node", "content", "clip", "video", "frame",
    "speaker", "emotion", "action", "event", "person", "object",
    "location", "scene", "time", "duration", "timestamp", "id",
    "type", "label", "tag", "category", "class", "data"
}


def is_meaningful_text(text: str) -> bool:
    """判断是否为真实的记忆内容，而不是短标签。"""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    if len(text) <= 10:
        return False
    
    line_count = text.count('\n')
    if line_count > 0:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        avg_len = sum(len(l) for l in lines) / len(lines) if lines else 0
        if avg_len < 8:
            return False
    
    if ' ' not in text:
        return False
    
    words = text.lower().split()
    if all(word in LABEL_KEYWORDS for word in words):
        return False
    
    if all(c.isdigit() or c in ' \n\t' for c in text):
        return False
    
    return True


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_text(value: Any) -> str:
    """Normalize value into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                t = item.strip()
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()
    return ""


def get_node_text(node: Any) -> str:
    """Extract meaningful text from a node object."""
    # 方式1：从 metadata.contents 中获取（推荐）
    if hasattr(node, "metadata"):
        try:
            metadata = getattr(node, "metadata")
            if isinstance(metadata, dict):
                contents = metadata.get("contents")
                if contents:
                    text = normalize_text(contents)
                    if text and is_meaningful_text(text):
                        return text
        except Exception:
            pass
    
    # 方式2：尝试直接的字段（备选）
    for field in TEXT_FIELD_CANDIDATES:
        if hasattr(node, field):
            try:
                val = getattr(node, field)
                text = normalize_text(val)
                if text and is_meaningful_text(text):
                    return text
            except Exception:
                continue
    
    return ""


def to_sorted_clip_ids(mapping: Any) -> list[Any]:
    if not isinstance(mapping, dict):
        return []
    try:
        return sorted(mapping.keys())
    except Exception:
        return sorted(mapping.keys(), key=lambda x: str(x))


def build_clip_docs(graph: Any) -> list[dict[str, Any]]:
    """Extract text and embeddings from VideoGraph, aggregated by clip."""
    nodes = getattr(graph, "nodes", None)
    text_nodes_by_clip = getattr(graph, "text_nodes_by_clip", None)

    if not isinstance(nodes, dict):
        raise ValueError("graph.nodes is missing or not a dict.")
    if not isinstance(text_nodes_by_clip, dict):
        raise ValueError("graph.text_nodes_by_clip is missing or not a dict.")

    clip_docs: list[dict[str, Any]] = []
    clip_ids = to_sorted_clip_ids(text_nodes_by_clip)

    for clip_id in clip_ids:
        raw_node_ids = text_nodes_by_clip.get(clip_id, [])
        node_ids: list[Any] = list(raw_node_ids) if isinstance(raw_node_ids, (list, tuple)) else []

        texts: list[str] = []
        kept_node_ids: list[Any] = []
        embeddings_list: list[np.ndarray] = []

        for nid in node_ids:
            node = nodes.get(nid)
            if node is None:
                continue
            text = get_node_text(node).strip()
            if text:
                texts.append(text)
                kept_node_ids.append(nid)
                
                # 提取和规范化 embeddings
                try:
                    if hasattr(node, "embeddings") and node.embeddings is not None:
                        emb = node.embeddings
                        # 转换为 numpy 数组（可能原本是 list）
                        if isinstance(emb, list):
                            emb = np.array(emb, dtype=np.float32)
                        elif isinstance(emb, np.ndarray):
                            emb = emb.astype(np.float32)
                        else:
                            continue
                        
                        # 如果是 (1, dim) 的形状，压平为 (dim,)
                        if emb.ndim == 2 and emb.shape[0] == 1:
                            emb = emb.squeeze(0)
                        
                        embeddings_list.append(emb)
                except Exception:
                    pass

        clip_text = "\n".join(texts).strip()
        
        # 聚合 embeddings：如果有多个节点，取平均值并归一化
        clip_embedding = None
        if embeddings_list:
            try:
                clip_embedding = np.mean(embeddings_list, axis=0, dtype=np.float32)
                # L2 归一化
                norm = np.linalg.norm(clip_embedding)
                if norm > 1e-10:
                    clip_embedding = clip_embedding / norm
            except Exception:
                pass

        doc_dict: dict[str, Any] = {
            "clip_id": clip_id,
            "node_ids": kept_node_ids,
            "num_text_nodes": len(kept_node_ids),
            "text": clip_text,
        }
        
        # 如果有 embedding，加入到字典中（转为列表以便 JSON 序列化）
        if clip_embedding is not None:
            doc_dict["embedding"] = clip_embedding.tolist()
            doc_dict["embedding_dim"] = int(clip_embedding.shape[0])
        
        clip_docs.append(doc_dict)

    return clip_docs


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def preview_text(text: str, max_chars: int = 240) -> str:
    s = " ".join(text.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clip_docs with embeddings from VideoGraph pickle.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to memory graph .pkl")
    parser.add_argument("--out", type=str, required=False, help="Output path (optional, defaults to {pkl_stem}_embeddings.json in pkl directory)")
    parser.add_argument("--out-dir", type=str, required=False, help="Output directory for embeddings file (defaults to pkl directory)")
    args = parser.parse_args()

    # 转换为绝对路径（相对于项目根目录）
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = _PROJECT_ROOT / pkl_path
    
    # 确定输出路径
    if args.out:
        # 如果明确指定了输出路径，使用该路径
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path
    else:
        # 否则根据pkl文件名自动生成输出文件名
        pkl_stem = pkl_path.stem  # 例如 "bedroom_01"
        out_filename = f"{pkl_stem}_embeddings.json"
        
        if args.out_dir:
            out_dir = Path(args.out_dir)
            if not out_dir.is_absolute():
                out_dir = _PROJECT_ROOT / out_dir
        else:
            # 默认在pkl所在目录中生成
            out_dir = pkl_path.parent
        
        out_path = out_dir / out_filename

    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        return 1

    try:
        graph = load_pickle(pkl_path)
    except Exception as e:
        print(f"ERROR: Failed to load pickle: {e}")
        return 1

    try:
        clip_docs = build_clip_docs(graph)
    except Exception as e:
        print(f"ERROR: Failed to build clip docs: {e}")
        return 1

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, clip_docs)
    except Exception as e:
        print(f"ERROR: Failed to save json: {e}")
        return 1

    total = len(clip_docs)
    empty = sum(1 for d in clip_docs if not str(d.get("text", "")).strip())
    with_embedding = sum(1 for d in clip_docs if "embedding" in d)

    print("=" * 80)
    print("Build clip_docs with embeddings completed")
    print("=" * 80)
    print(f"Input pkl: {pkl_path}")
    print(f"Output file: {out_path}")
    print(f"Total clip docs: {total}")
    print(f"Empty clips: {empty}")
    print(f"Clips with embeddings: {with_embedding}")

    previews = clip_docs[:3]
    for i, item in enumerate(previews, 1):
        has_emb_str = f"✓ {item.get('embedding_dim', 0)}d" if "embedding" in item else "✗ no embedding"
        print(f"\nPreview #{i} [{has_emb_str}]")
        print(f"  clip_id: {item.get('clip_id')}")
        print(f"  num_text_nodes: {item.get('num_text_nodes')}")
        print(f"  node_ids(head): {list(item.get('node_ids', []))[:10]}")
        print(f"  text: {preview_text(str(item.get('text', '')))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
