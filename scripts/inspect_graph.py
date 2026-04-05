#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a VideoGraph pickle file and print its internal structure.

Usage:
  python inspect_graph.py --pkl path/to/memory_graph.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

# 添加项目根目录到Python搜索路径，以便能导入mmagent模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def safe_len(obj: Any) -> int | None:
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:
        return None


def safe_keys(obj: Any, n: int = 5) -> list[Any]:
    try:
        if isinstance(obj, dict):
            return list(obj.keys())[:n]
        if isinstance(obj, (list, tuple)):
            return list(obj)[:n]
        if hasattr(obj, "keys"):
            return list(obj.keys())[:n]  # type: ignore[attr-defined]
    except Exception:
        return []
    return []


def short_repr(obj: Any, max_len: int = 800) -> str:
    text = repr(obj)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def iter_first(items: Iterable[Any]) -> Any | None:
    for x in items:
        return x
    return None


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def inspect_graph(graph: Any) -> None:
    print("=" * 80)
    print("VideoGraph Inspect Result")
    print("=" * 80)

    print(f"[1] Graph object type: {type(graph)}")

    fields = ["nodes", "text_nodes", "text_nodes_by_clip", "edges", "event_sequence_by_clip"]
    print("\n[2] Field existence")
    for name in fields:
        exists = hasattr(graph, name)
        print(f"  - {name}: {'YES' if exists else 'NO'}")

    nodes = getattr(graph, "nodes", None)
    text_nodes = getattr(graph, "text_nodes", None)
    text_nodes_by_clip = getattr(graph, "text_nodes_by_clip", None)
    edges = getattr(graph, "edges", None)
    event_sequence_by_clip = getattr(graph, "event_sequence_by_clip", None)

    print("\n[3] Basic counts")
    print(f"  - nodes total: {safe_len(nodes)}")
    print(f"  - text_nodes total: {safe_len(text_nodes)}")
    print(f"  - edges total: {safe_len(edges)}")
    print(f"  - clips in text_nodes_by_clip: {safe_len(text_nodes_by_clip)}")
    print(f"  - clips in event_sequence_by_clip: {safe_len(event_sequence_by_clip)}")

    print("\n[4] Clip preview")
    clip_ids: list[Any] = []
    if isinstance(text_nodes_by_clip, dict):
        try:
            clip_ids = sorted(text_nodes_by_clip.keys(), key=lambda x: str(x))
        except Exception:
            clip_ids = list(text_nodes_by_clip.keys())
    elif text_nodes_by_clip is not None and hasattr(text_nodes_by_clip, "keys"):
        try:
            clip_ids = list(text_nodes_by_clip.keys())  # type: ignore[attr-defined]
        except Exception:
            clip_ids = []

    print(f"  - first clip ids (up to 10): {clip_ids[:10]}")

    sample_clip = clip_ids[0] if clip_ids else None
    if sample_clip is not None:
        try:
            sample_node_ids = text_nodes_by_clip[sample_clip]  # type: ignore[index]
            print(f"  - sample clip_id: {sample_clip}")
            print(f"  - text node ids in sample clip (up to 20): {list(sample_node_ids)[:20]}")
        except Exception as e:
            print(f"  - failed to read sample clip nodes: {e}")

    print("\n[5] Node sample")
    sample_node = None
    sample_node_id = None
    if isinstance(nodes, dict) and nodes:
        sample_node_id = iter_first(nodes.keys())
        try:
            sample_node = nodes[sample_node_id]
        except Exception:
            sample_node = None
    elif nodes is not None and hasattr(nodes, "items"):
        try:
            items = list(nodes.items())  # type: ignore[attr-defined]
            if items:
                sample_node_id, sample_node = items[0]
        except Exception:
            pass

    if sample_node is None:
        print("  - no sample node available.")
    else:
        print(f"  - sample node id: {sample_node_id}")
        print(f"  - sample node type: {type(sample_node)}")
        attr_names = [x for x in dir(sample_node) if not x.startswith("__")]
        print(f"  - sample node attributes (first 40): {attr_names[:40]}")
        if hasattr(sample_node, "__dict__"):
            try:
                d = getattr(sample_node, "__dict__")
                print(f"  - sample node __dict__: {short_repr(d)}")
            except Exception as e:
                print(f"  - failed to print __dict__: {e}")
        else:
            print("  - sample node has no __dict__.")

    print("\n[6] Raw previews")
    print(f"  - text_nodes_by_clip keys preview: {safe_keys(text_nodes_by_clip, 8)}")
    print(f"  - event_sequence_by_clip keys preview: {safe_keys(event_sequence_by_clip, 8)}")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect VideoGraph pickle structure.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to memory graph .pkl file")
    args = parser.parse_args()

    # 改变工作目录到项目根目录，确保相对路径能正确解析
    project_root = Path(__file__).parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    # 处理pickle路径：如果是相对路径，相对于项目根；如果是绝对路径，直接使用
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = project_root / pkl_path
    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        return 1

    try:
        graph = load_pickle(pkl_path)
    except Exception as e:
        print(f"ERROR: Failed to load pickle: {e}")
        return 1

    try:
        inspect_graph(graph)
    except Exception as e:
        print(f"ERROR: Failed to inspect graph: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
