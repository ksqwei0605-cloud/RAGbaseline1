#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation helper for baseline1 RAG system.

This script:
1. Loads benchmark annotations (supports nested robot.json format)
2. Filters samples by target pkl file
3. Runs QA evaluation by calling the retrieval + chat pipeline
4. Saves results to a JSON file for analysis

Supported annotation formats:
- Flat list: [{"question": "...", "answer": "...", "mem_path": "..."}, ...]
- Nested (robot.json): {"scene_name": {"mem_path": "...", "qa_list": [...]}, ...}

Usage:
  python run_baseline1_eval.py \\
    --annotation_file data/annotations/robot.json \\
    --target_pkl data/memory_graphs/robot/living_room_06.pkl \\
    --clip_docs_dir baseline1/outputs \\
    --output_file baseline1/outputs/eval_results.json \\
    --api_key YOUR_API_KEY \\
    --base_url https://api.example.com/v1 \\
    --chat_model doubao-seed-2.0lite \\
    --top_k 3 \\
    --backend precomputed

Environment variables (as fallback):
  - OPENAI_API_KEY: API key
  - OPENAI_BASE_URL: API base URL
  - OPENAI_CHAT_MODEL: Chat model name
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("question", "answer", "mem_path")


def load_annotations(annotation_file: str) -> list[dict[str, Any]]:
    path = Path(annotation_file)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in annotation file {path}: {e}") from e

    validated_samples: list[dict[str, Any]] = []

    # 支持两种格式：
    # 1. 列表格式（平坦）：[{question, answer, mem_path, ...}, ...]
    # 2. 嵌套格式（robot.json）：{scene_name: {mem_path, qa_list: [...]}, ...}
    
    if isinstance(data, list):
        # 格式1：平坦列表
        for index, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Sample at index {index} must be a JSON object, "
                    f"got {type(sample).__name__}."
                )
            validate_sample_fields(sample, index=index)
            validated_samples.append(sample)
    
    elif isinstance(data, dict):
        # 格式2：嵌套结构（robot.json）
        for scene_name, scene_data in data.items():
            if not isinstance(scene_data, dict):
                raise ValueError(f"Scene '{scene_name}' must be a JSON object, got {type(scene_data).__name__}.")
            
            mem_path = scene_data.get("mem_path")
            if not mem_path:
                raise ValueError(f"Scene '{scene_name}' missing 'mem_path' field.")
            
            qa_list = scene_data.get("qa_list", [])
            if not isinstance(qa_list, list):
                raise ValueError(f"Scene '{scene_name}' qa_list must be a list, got {type(qa_list).__name__}.")
            
            for qa_index, qa_item in enumerate(qa_list):
                if not isinstance(qa_item, dict):
                    raise ValueError(
                        f"Scene '{scene_name}' qa_list[{qa_index}] must be a JSON object, "
                        f"got {type(qa_item).__name__}."
                    )
                
                # 展平为平坦结构
                sample = {
                    "question": qa_item.get("question"),
                    "answer": qa_item.get("answer"),
                    "mem_path": mem_path,
                }
                
                # 保留其他可选字段
                if "question_id" in qa_item:
                    sample["id"] = qa_item["question_id"]
                if "reasoning" in qa_item:
                    sample["reasoning"] = qa_item["reasoning"]
                if "timestamp" in qa_item:
                    sample["timestamp"] = qa_item["timestamp"]
                if "type" in qa_item:
                    sample["type"] = qa_item["type"]
                
                validate_sample_fields(sample, index=f"{scene_name}[{qa_index}]")
                validated_samples.append(sample)
    
    else:
        raise ValueError(
            f"Annotation file must be a JSON list or dict at top level. "
            f"Got {type(data).__name__} instead."
        )

    return validated_samples


def validate_sample_fields(sample: dict[str, Any], index: int | None = None) -> None:
    missing_fields = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing_fields:
        location = f" at index {index}" if index is not None else ""
        raise KeyError(
            f"Sample{location} is missing required field(s): {', '.join(missing_fields)}"
        )


def normalize_path_text(path_text: str) -> str:
    return Path(path_text).as_posix()


def match_mem_path(sample_mem_path: str, target_pkl: str) -> bool:
    sample_path = Path(sample_mem_path)
    target_path = Path(target_pkl)

    sample_norm = normalize_path_text(sample_mem_path)
    target_norm = normalize_path_text(target_pkl)

    return sample_norm == target_norm or sample_path.name == target_path.name


def filter_samples_by_pkl(samples: list[dict[str, Any]], target_pkl: str) -> list[dict[str, Any]]:
    matched_samples: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        validate_sample_fields(sample, index=index)

        mem_path = sample["mem_path"]
        if not isinstance(mem_path, str) or not mem_path.strip():
            raise ValueError(f"Sample at index {index} has invalid mem_path: {mem_path!r}")

        if match_mem_path(mem_path, target_pkl):
            matched_samples.append(sample)

    return matched_samples


def infer_clip_docs_path(mem_path: str, clip_docs_dir: str) -> str:
    """
    根据 mem_path 推断对应的 clip_docs_with_embeddings.json 路径
    
    例如：
    mem_path = "data/memory_graphs/robot/living_room_06.pkl"
    返回："{clip_docs_dir}/living_room_06_embeddings.json"
    """
    pkl_name = Path(mem_path).stem  # 提取文件名（不含后缀）
    clip_docs_name = f"{pkl_name}_embeddings.json"
    return str(Path(clip_docs_dir) / clip_docs_name)


def run_module4(
    question: str,
    mem_path: str,
    clip_docs_dir: str,
    api_key: str,
    base_url: str,
    chat_model: str,
    top_k: int = 3,
    backend: str = "precomputed",
) -> str:
    """
    调用 qa_once 核心逻辑获取答案
    
    参数：
    - question: 用户问题
    - mem_path: memory graph 的 pkl 路径（用于日志/调试）
    - clip_docs_dir: 包含 clip_docs_with_embeddings.json 的目录
    - api_key: OpenAI 兼容 API 密钥
    - base_url: API 基础 URL
    - chat_model: 聊天模型名称（如 doubao-seed-2.0lite）
    - top_k: 检索的 top-k clips 数量
    - backend: 检索后端（precomputed, tfidf, embedding_api）
    
    返回：
    - 模型生成的答案字符串
    """
    try:
        from retrieval import build_searcher, RetrievalResult
    except ImportError as e:
        raise ImportError("Failed to import retrieval module. Make sure it's in sys.path.") from e
    
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package is required. Install it with: pip install openai") from e
    
    # 推断对应的 clip_docs 文件
    clip_docs_path = infer_clip_docs_path(mem_path, clip_docs_dir)
    clip_docs_file = Path(clip_docs_path)
    
    if not clip_docs_file.exists():
        raise FileNotFoundError(
            f"Clip docs not found: {clip_docs_path}\n"
            f"Make sure to run build_clip_docs.py first for {mem_path}"
        )
    
    # 1️⃣ 初始化检索器
    try:
        retriever = build_searcher(
            clip_docs_path=str(clip_docs_path),
            backend=backend,
            api_key=api_key,
            base_url=base_url,
            embedding_model=None,  # precomputed 不需要
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize retriever: {e}") from e
    
    # 2️⃣ 检索相关 clips
    try:
        retrieved_clips = retriever.search_clip(question, top_k=top_k)
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}") from e
    
    if not retrieved_clips:
        raise RuntimeError("No clips retrieved for the question")
    
    # 3️⃣ 构建 context
    context_parts = []
    for item in retrieved_clips:
        context_parts.append(f"[Clip {item.clip_id}]\n{item.text.strip()}")
    context = "\n\n".join(context_parts).strip()
    
    # 4️⃣ 调用 Chat API
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        system_prompt = (
            "You are an assistant answering questions about a long-video memory graph.\n"
            "Use only the retrieved memory snippets as evidence.\n"
            "Prefer short, direct answers.\n"
            "If the answer is not explicitly stated, make only minimal and reasonable inference.\n"
            "Do not fabricate unsupported details."
        )
        
        user_prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Retrieved memory context:\n{context}\n\n"
            "Answer based only on the context above."
        )
        
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        answer = resp.choices[0].message.content.strip()
        if not answer:
            raise RuntimeError("Empty response from Chat API")
        
        return answer
    
    except Exception as e:
        raise RuntimeError(f"Chat API request failed: {e}") from e


def build_result_record(sample: dict[str, Any], model_answer: str, status: str) -> dict[str, Any]:
    result = {
        "mem_path": sample["mem_path"],
        "question": sample["question"],
        "ground_truth_answer": sample["answer"],
        "model_answer": model_answer,
        "status": status,
    }

    # 保留其他可选字段（question_id, reasoning, timestamp, type 等）
    if "id" in sample:
        result["id"] = sample["id"]
    if "reasoning" in sample:
        result["reasoning"] = sample["reasoning"]
    if "timestamp" in sample:
        result["timestamp"] = sample["timestamp"]
    if "type" in sample:
        result["type"] = sample["type"]

    return result


def save_results(output_file: str, results: list[dict[str, Any]]) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load benchmark annotations, run QA evaluation, and save results."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="data/annotations/robot.json",
        help="Path to annotation JSON file (supports nested robot.json format). Default: data/annotations/robot.json",
    )
    parser.add_argument(
        "--target_pkl",
        type=str,
        required=True,
        help="Target pkl path or filename used to filter samples.",
    )
    parser.add_argument(
        "--clip_docs_dir",
        type=str,
        required=True,
        help="Directory containing clip_docs_with_embeddings.json files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSON results file.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI-compatible API key. Can also use OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL. Can also use OPENAI_BASE_URL env var.",
    )
    parser.add_argument(
        "--chat_model",
        type=str,
        default=None,
        help="Chat model name (e.g., doubao-seed-2.0lite). Can also use OPENAI_CHAT_MODEL env var.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of clips to retrieve. Default: 3",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="precomputed",
        choices=["precomputed", "tfidf", "embedding_api"],
        help="Retrieval backend. Default: precomputed",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 从环境变量或命令行参数获取 API 配置
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    chat_model = args.chat_model or os.getenv("OPENAI_CHAT_MODEL", "")
    
    # 验证必需参数
    if not api_key:
        print("ERROR: Missing API key. Set --api_key or OPENAI_API_KEY environment variable.", file=sys.stderr)
        return 1
    if not chat_model:
        print("ERROR: Missing chat model. Set --chat_model or OPENAI_CHAT_MODEL environment variable.", file=sys.stderr)
        return 1
    
    # 验证 clip_docs_dir 存在
    clip_docs_dir = Path(args.clip_docs_dir)
    if not clip_docs_dir.is_dir():
        print(f"ERROR: clip_docs_dir not found: {clip_docs_dir}", file=sys.stderr)
        return 1

    try:
        samples = load_annotations(args.annotation_file)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        matched_samples = filter_samples_by_pkl(samples, args.target_pkl)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not matched_samples:
        print(
            "No samples matched target_pkl: "
            f"{args.target_pkl}. Nothing will be written."
        )
        return 0

    results: list[dict[str, Any]] = []
    total_count = len(matched_samples)

    print(f"Starting evaluation with {total_count} samples...")
    print(f"  Backend: {args.backend}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Chat model: {chat_model}")
    print()

    for index, sample in enumerate(matched_samples, start=1):
        question = sample["question"]
        mem_path = sample["mem_path"]
        sample_id = sample.get("id", f"sample_{index}")

        print(f"[{index}/{total_count}] {sample_id}")
        print(f"  Question: {question[:80]}..." if len(question) > 80 else f"  Question: {question}")

        try:
            model_answer = run_module4(
                question=question,
                mem_path=mem_path,
                clip_docs_dir=str(clip_docs_dir),
                api_key=api_key,
                base_url=base_url,
                chat_model=chat_model,
                top_k=args.top_k,
                backend=args.backend,
            )
            record = build_result_record(sample, model_answer=model_answer, status="ok")
            print(f"  Answer: {model_answer[:80]}..." if len(model_answer) > 80 else f"  Answer: {model_answer}")
            print(f"  Status: ✓ OK")
        except Exception as e:
            error_message = str(e)
            record = build_result_record(sample, model_answer=error_message, status="error")
            print(f"  Status: ✗ FAILED")
            print(f"  Error: {error_message[:100]}...")

        results.append(record)
        print()

    try:
        save_results(args.output_file, results)
    except OSError as e:
        print(f"ERROR: Failed to write output file {args.output_file}: {e}", file=sys.stderr)
        return 1

    # 统计结果
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print("=" * 80)
    print(f"Evaluation completed!")
    print(f"  Saved {len(results)} result(s) to: {args.output_file}")
    print(f"  Success: {ok_count}/{total_count}")
    print(f"  Failed: {error_count}/{total_count}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
