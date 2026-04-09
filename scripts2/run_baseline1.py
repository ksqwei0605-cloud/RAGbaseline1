#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个文件的主要功能：
对指定 pkl 对应的全部问题运行 character-aware baseline1，
自动准备 docs/vectors，并将问题、标准答案、预测答案及中间检索结果保存为结果 json。

说明：
- 支持传入 pkl_name 或 pkl_path。
- 会从 robot.json 中尽量稳健地精确匹配对应问题。
- 最终输出到 outputs/{pkl_name}_baseline1_results.json。
- answer_model 走 OpenAI-compatible chat 接口，embedding_model 走 Ark SDK。
- 这是 character-aware baseline1 的集成运行脚本。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from build_clip_docs import extract_clip_docs, load_video_graph, save_docs_json
from qa_once import answer_once


LOGGER = logging.getLogger("run_baseline1")


def setup_logging() -> None:
    """初始化日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(value: Any) -> str:
    """
    归一化名称，便于在 robot.json 中做稳健匹配。

    策略：
    - 若是路径，取 basename
    - 去掉扩展名
    - 统一小写
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    basename = Path(text).name
    stem = Path(basename).stem
    return stem.lower()


def resolve_pkl_path(pkl_name: str, pkl_path: str, memory_graph_dir: str) -> Path:
    """
    根据 pkl_name 或 pkl_path 找到目标 pkl 文件。
    """
    if pkl_path:
        path = Path(pkl_path)
        if not path.is_file():
            raise FileNotFoundError("Specified pkl_path not found: {0}".format(path))
        return path

    if not pkl_name:
        raise ValueError("Either pkl_name or pkl_path must be provided.")

    memory_graph_root = Path(memory_graph_dir)
    if not memory_graph_root.is_dir():
        raise FileNotFoundError("memory_graph_dir not found: {0}".format(memory_graph_root))

    candidates = list(memory_graph_root.rglob("{0}.pkl".format(pkl_name)))
    if not candidates:
        raise FileNotFoundError("Cannot find pkl named {0}.pkl under {1}".format(pkl_name, memory_graph_root))
    if len(candidates) > 1:
        LOGGER.warning("Found multiple pkl candidates for %s. Using the first one: %s", pkl_name, candidates[0])
    return candidates[0]


def ensure_docs_exist(pkl_path: str, output_dir: str) -> str:
    """
    确保 docs.json 存在；若不存在则从 pkl 构建。
    """
    pkl_file = Path(pkl_path)
    pkl_name = pkl_file.stem
    docs_path = Path(output_dir) / "{0}_docs.json".format(pkl_name)
    if docs_path.is_file():
        LOGGER.info("Docs already exist: %s", docs_path)
        return str(docs_path)

    LOGGER.info("Docs not found. Building docs for %s", pkl_file)
    graph = load_video_graph(str(pkl_file))
    docs_data = extract_clip_docs(graph, pkl_name=pkl_name)
    save_docs_json(docs_data, str(docs_path))
    return str(docs_path)


def load_robot_annotations(annotations_dir: str) -> Dict[str, Any]:
    """读取 annotations_dir 下的 robot.json。"""
    robot_path = Path(annotations_dir) / "robot.json"
    if not robot_path.is_file():
        raise FileNotFoundError("robot.json not found: {0}".format(robot_path))

    LOGGER.info("Loading robot annotations from %s", robot_path)
    with robot_path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("robot.json top-level structure must be a dict.")
    return data


def match_questions_for_pkl(robot_data: Dict[str, Any], pkl_name: str, pkl_path: str) -> List[Dict[str, Any]]:
    """
    从 robot.json 中精确找到该 pkl_name 对应的问题。

    匹配策略说明：
    - 先尝试顶层 key 是否直接等于 pkl_name。
    - 若不命中，再尝试 scene 数据中的常见字段：
      mem_path / video_path / pkl_name / video_id / scene_name 等。
    - 匹配时统一做 basename + stem + lowercase 归一化。
    """
    normalized_target = normalize_name(pkl_name)
    normalized_target_path = normalize_name(pkl_path)
    exact_matches: List[Dict[str, Any]] = []

    for scene_key, scene_value in robot_data.items():
        if not isinstance(scene_value, dict):
            continue

        candidate_names = {
            normalize_name(scene_key),
            normalize_name(scene_value.get("pkl_name")),
            normalize_name(scene_value.get("video_id")),
            normalize_name(scene_value.get("scene_name")),
            normalize_name(scene_value.get("mem_path")),
            normalize_name(scene_value.get("video_path")),
        }

        if normalized_target not in candidate_names and normalized_target_path not in candidate_names:
            continue

        qa_list = scene_value.get("qa_list", [])
        if not isinstance(qa_list, list):
            LOGGER.warning("Scene %s has invalid qa_list, skip.", scene_key)
            continue

        for index, qa_item in enumerate(qa_list):
            if not isinstance(qa_item, dict):
                LOGGER.warning("Scene %s qa_list[%s] is not a dict, skip.", scene_key, index)
                continue

            exact_matches.append(
                {
                    "question_id": qa_item.get("question_id", "{0}_Q{1:02d}".format(normalized_target, index + 1)),
                    "pkl_name": normalized_target,
                    "question": qa_item.get("question", ""),
                    "gold_answer": qa_item.get("answer", ""),
                    "raw_item": qa_item,
                }
            )

    if not exact_matches:
        raise ValueError("No questions matched pkl_name={0} pkl_path={1}".format(pkl_name, pkl_path))

    LOGGER.info("Matched %s questions for pkl_name=%s", len(exact_matches), normalized_target)
    return exact_matches


def run_single_question(
    sample: Dict[str, Any],
    docs_path: str,
    vectors_path: str,
    answer_model: str,
    embedding_model: str,
    answer_api_key: str,
    answer_base_url: str = "",
    ark_api_key: str = "",
    top_k_clips: int = 3,
    embedding_batch_size: int = 64,
) -> Dict[str, Any]:
    """运行单条问题。"""
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("Sample question is empty.")

    LOGGER.info("Running question_id=%s", sample.get("question_id"))
    qa_result = answer_once(
        question=question,
        docs_path=docs_path,
        vectors_path=vectors_path,
        answer_model=answer_model,
        embedding_model=embedding_model,
        answer_api_key=answer_api_key,
        answer_base_url=answer_base_url,
        ark_api_key=ark_api_key,
        top_k_clips=top_k_clips,
        embedding_batch_size=embedding_batch_size,
    )

    return {
        "question_id": sample.get("question_id"),
        "pkl_name": sample.get("pkl_name"),
        "question": question,
        "gold_answer": sample.get("gold_answer", ""),
        "pred_answer": qa_result["pred_answer"],
        "field": qa_result["field"],
        "retrieved_clips": qa_result["retrieved_clips"],
        "retrieved_nodes": qa_result["retrieved_nodes"],
        "context": qa_result["context"],
        "character_hints": qa_result["character_hints"],
        "classifier_raw_output": qa_result["classifier_raw_output"],
        "embedding_model": qa_result["embedding_model"],
        "answer_model": qa_result["answer_model"],
    }


def save_results(results: Dict[str, Any], out_path: str) -> None:
    """保存最终结果 json。"""
    path = Path(out_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("Saved baseline1 results to %s", path)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run baseline1 for all questions of one pkl.")
    parser.add_argument("--pkl_name", default="", help="PKL stem name, for example bedroom_01.")
    parser.add_argument("--pkl_path", default="", help="Direct path to the target pkl file.")
    parser.add_argument(
        "--memory_graph_dir",
        default=str(PROJECT_ROOT / "data" / "memory_graphs"),
        help="Directory containing pkl files.",
    )
    parser.add_argument(
        "--annotations_dir",
        default=str(PROJECT_ROOT / "data" / "annotations"),
        help="Directory containing robot.json.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "baseline1" / "outputs"),
        help="Output directory for docs, vectors, and results.",
    )
    parser.add_argument("--answer_model", required=True, help="Answer model for classification and QA.")
    parser.add_argument("--embedding_model", required=True, help="Embedding model for docs/query vectorization.")
    parser.add_argument(
        "--answer_api_key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key for answer_model chat requests.",
    )
    parser.add_argument(
        "--answer_base_url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible base url for answer_model chat requests.",
    )
    parser.add_argument(
        "--ark_api_key",
        default=os.getenv("ARK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        help="Ark API key for embedding requests.",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Legacy alias. If provided, it will be used as default for both answer_api_key and ark_api_key.",
    )
    parser.add_argument(
        "--base_url",
        default="",
        help="Legacy alias. If provided, it will be used as default for answer_base_url.",
    )
    parser.add_argument("--top_k_clips", type=int, default=3, help="Top-k clips for retrieval.")
    parser.add_argument("--embedding_batch_size", type=int, default=64, help="Batch size for vector building.")
    return parser.parse_args()


def resolve_runtime_args(args: argparse.Namespace) -> Dict[str, str]:
    """统一解析新旧参数，尽量兼容旧命令。"""
    answer_api_key = args.answer_api_key or args.api_key
    answer_base_url = args.answer_base_url or args.base_url
    ark_api_key = args.ark_api_key or args.api_key
    return {
        "answer_api_key": answer_api_key,
        "answer_base_url": answer_base_url,
        "ark_api_key": ark_api_key,
    }


def main() -> int:
    """命令行入口。"""
    setup_logging()
    args = parse_args()
    runtime_args = resolve_runtime_args(args)

    try:
        pkl_file = resolve_pkl_path(args.pkl_name, args.pkl_path, args.memory_graph_dir)
        pkl_name = pkl_file.stem
        output_dir = Path(args.output_dir)
        ensure_dir(output_dir)

        docs_path = ensure_docs_exist(str(pkl_file), str(output_dir))
        vectors_path = str(output_dir / "{0}_vectors.npz".format(pkl_name))

        robot_data = load_robot_annotations(args.annotations_dir)
        matched_samples = match_questions_for_pkl(robot_data, pkl_name=pkl_name, pkl_path=str(pkl_file))

        result_items: List[Dict[str, Any]] = []
        for sample in matched_samples:
            result_item = run_single_question(
                sample=sample,
                docs_path=docs_path,
                vectors_path=vectors_path,
                answer_model=args.answer_model,
                embedding_model=args.embedding_model,
                answer_api_key=runtime_args["answer_api_key"],
                answer_base_url=runtime_args["answer_base_url"],
                ark_api_key=runtime_args["ark_api_key"],
                top_k_clips=args.top_k_clips,
                embedding_batch_size=args.embedding_batch_size,
            )
            result_items.append(result_item)

        results_payload = {
            "pkl_name": pkl_name,
            "pkl_path": str(pkl_file),
            "docs_path": docs_path,
            "vectors_path": vectors_path,
            "num_questions": len(result_items),
            "embedding_model": args.embedding_model,
            "answer_model": args.answer_model,
            "results": result_items,
        }

        results_path = output_dir / "{0}_baseline1_results.json".format(pkl_name)
        save_results(results_payload, str(results_path))
    except Exception as exc:
        LOGGER.exception("run_baseline1 failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
