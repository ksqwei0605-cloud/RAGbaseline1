#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个文件的主要功能：
执行一次 baseline1 问答，包括问题分类、query embedding、向量文件检查、
检索上下文，以及基于上下文调用 answer_model 回答。

说明：
- answer_model 负责分类和回答，走 OpenAI-compatible chat 接口。
- embedding_model 负责 docs/query 向量化，走火山方舟 Ark SDK。
- 若 vectors.npz 不存在，会自动调用 build_clip_embeddings.py 对 docs.json 建向量。
- qa_once 现在是 character-aware 回答流程，重点增强的是 answer prompt，而不是重写 retrieval 主体。
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from build_clip_embeddings import build_clip_embeddings, create_ark_client, get_single_embedding_from_ark
from retrieval import retrieve


LOGGER = logging.getLogger("qa_once")
VALID_FIELDS = {"episodic", "semantic", "all"}


def setup_logging() -> None:
    """初始化日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_openai_client(api_key: str, base_url: str = "") -> Any:
    """创建 OpenAI 兼容客户端。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for answer_model calls.") from exc

    if not api_key:
        raise ValueError("Missing API key for answer_model requests.")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def build_classification_prompt(question: str) -> List[Dict[str, str]]:
    """
    构造问题分类 prompt。

    输入：
    - question: 用户问题

    输出：
    - chat messages
    """
    system_prompt = (
        "You are a classifier for a video-memory QA system.\n"
        "Classify the question into exactly one field: episodic, semantic, or all.\n"
        "Return JSON only, for example: {\"field\": \"episodic\"}.\n"
        "Use 'all' if the question is mixed, unclear, or classification is uncertain."
    )
    user_prompt = "Question:\n{0}".format(question.strip())
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_candidate(text: str) -> str:
    """从模型输出中尽量提取 JSON 片段。"""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        return match.group(0)
    return stripped


def parse_field_from_classifier_output(raw_output: str) -> str:
    """
    从分类模型输出中解析字段。

    输入：
    - raw_output: answer_model 原始输出

    输出：
    - episodic / semantic / all
    """
    if not raw_output:
        return "all"

    candidate = extract_json_candidate(raw_output)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            field = str(parsed.get("field", "")).strip().lower()
            if field in VALID_FIELDS:
                return field
    except Exception:
        pass

    lowered = raw_output.strip().lower()
    for field in ("episodic", "semantic", "all"):
        if re.search(r"\b{0}\b".format(field), lowered):
            return field
    return "all"


def build_answer_prompt(question: str, context: str, character_hints: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    构造 character-aware 回答 prompt。

    重点要求：
    - 先做角色理解，再回答
    - 若 raw / normalized 同时出现，优先参考 normalized
    - 不要因为题目里出现 Lily / Emma 等名字但上下文里只出现内部标签，就过早回答 uncertain
    """
    system_prompt = (
        "You answer questions about a video memory graph.\n"
        "The context may contain internal entity tags such as <face_x>, <voice_x>, and <character_x>.\n"
        "These tags may refer to the same real person or role.\n"
        "If a node shows both raw text and normalized text, prefer the normalized text for character-level reasoning.\n"
        "Do not answer 'uncertain' merely because the literal name in the question (for example Lily or Emma) "
        "does not appear verbatim in the context.\n"
        "Instead, first infer which internal tags or normalized character tags may correspond to the same role, "
        "then answer using the best-supported evidence.\n"
        "Only answer 'uncertain' if the context truly contains no relevant evidence, or the evidence is far too weak "
        "to support any reasonable answer.\n"
        "Your response must have exactly two sections:\n"
        "Relevant evidence / alias interpretation:\n"
        "Final answer:"
    )

    character_hints_text = json.dumps(character_hints, ensure_ascii=False, indent=2)
    safe_context = context.strip() if context.strip() else "(empty)"
    user_prompt = (
        "Question:\n{0}\n\n"
        "Character hints:\n{1}\n\n"
        "Retrieved context:\n{2}\n\n"
        "Instructions:\n"
        "1. In 'Relevant evidence / alias interpretation', explain which evidence lines matter and whether any internal tags "
        "or normalized character tags may refer to the same role.\n"
        "2. In 'Final answer', provide the best-supported answer to the question.\n"
        "3. If the evidence is partial, give the best-supported answer instead of refusing too early.\n"
        "4. Use 'uncertain' only when the context has no relevant evidence or is truly insufficient for any reasonable answer."
    ).format(question.strip(), character_hints_text, safe_context)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def chat_completion(
    messages: List[Dict[str, str]],
    answer_model: str,
    answer_api_key: str,
    answer_base_url: str = "",
    temperature: float = 0.0,
) -> str:
    """调用 answer_model 进行 chat completion。"""
    client = create_openai_client(api_key=answer_api_key, base_url=answer_base_url)
    response = client.chat.completions.create(
        model=answer_model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return str(content).strip() if content else ""


def get_query_embedding(
    question: str,
    embedding_model: str,
    ark_api_key: str,
) -> np.ndarray:
    """
    使用 Ark SDK 为问题生成向量。

    输入：
    - question: 用户问题
    - embedding_model: embedding 模型名
    - ark_api_key: Ark API key

    输出：
    - shape=(dim,) 的 query embedding
    """
    client = create_ark_client(ark_api_key=ark_api_key)
    vector = get_single_embedding_from_ark(
        client=client,
        text=question.strip(),
        embedding_model=embedding_model,
    )
    query_embedding = np.asarray(vector, dtype=np.float32)
    LOGGER.info("Generated query embedding with dim=%s", query_embedding.shape[0])
    return query_embedding


def ensure_vectors_exist(
    docs_path: str,
    vectors_path: str,
    embedding_model: str,
    ark_api_key: str,
    batch_size: int = 64,
) -> str:
    """
    确保 vectors.npz 存在；若不存在则自动构建。

    输入：
    - docs_path: docs json 路径
    - vectors_path: 目标 vectors npz 路径

    输出：
    - 可用的 vectors 路径
    """
    vectors_file = Path(vectors_path)
    if vectors_file.is_file():
        LOGGER.info("Vectors already exist: %s", vectors_file)
        return str(vectors_file)

    LOGGER.info("Vectors not found. Building embeddings from docs: %s", docs_path)
    built_path = build_clip_embeddings(
        docs_path=docs_path,
        output_dir=str(vectors_file.parent),
        embedding_model=embedding_model,
        ark_api_key=ark_api_key,
        batch_size=batch_size,
    )

    if not Path(built_path).is_file():
        raise FileNotFoundError("Failed to build vectors file: {0}".format(built_path))
    return built_path


def answer_once(
    question: str,
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
    """
    执行一次完整 baseline1 QA。

    主流程保持不变：
    - 分类
    - fallback 到 all
    - query embedding
    - retrieval
    - answer_model 回答
    """
    LOGGER.info("Classifying question: %s", question)
    classifier_raw_output = chat_completion(
        messages=build_classification_prompt(question),
        answer_model=answer_model,
        answer_api_key=answer_api_key,
        answer_base_url=answer_base_url,
        temperature=0.0,
    )
    field = parse_field_from_classifier_output(classifier_raw_output)
    LOGGER.info("Classifier field=%s raw_output=%s", field, classifier_raw_output)

    vectors_path = ensure_vectors_exist(
        docs_path=docs_path,
        vectors_path=vectors_path,
        embedding_model=embedding_model,
        ark_api_key=ark_api_key,
        batch_size=embedding_batch_size,
    )

    query_embedding = get_query_embedding(
        question=question,
        embedding_model=embedding_model,
        ark_api_key=ark_api_key,
    )

    retrieval_result = retrieve(
        field=field,
        query_embedding=query_embedding,
        docs_path=docs_path,
        vectors_path=vectors_path,
        top_k_clips=top_k_clips,
    )
    effective_field = field

    if not retrieval_result["retrieved_clips"] and field != "all":
        LOGGER.info("No retrieval hits for field=%s. Falling back to field=all.", field)
        retrieval_result = retrieve(
            field="all",
            query_embedding=query_embedding,
            docs_path=docs_path,
            vectors_path=vectors_path,
            top_k_clips=top_k_clips,
        )
        effective_field = "all"

    LOGGER.info("Retrieved top-%s clips for QA.", len(retrieval_result["retrieved_clips"]))

    pred_answer = chat_completion(
        messages=build_answer_prompt(
            question=question,
            context=retrieval_result["context"],
            character_hints=retrieval_result["character_hints"],
        ),
        answer_model=answer_model,
        answer_api_key=answer_api_key,
        answer_base_url=answer_base_url,
        temperature=0.0,
    )
    LOGGER.info("Generated answer: %s", pred_answer)

    return {
        "pred_answer": pred_answer,
        "field": effective_field,
        "retrieved_clips": retrieval_result["retrieved_clips"],
        "retrieved_nodes": retrieval_result["retrieved_nodes"],
        "context": retrieval_result["context"],
        "character_hints": retrieval_result["character_hints"],
        "classifier_raw_output": classifier_raw_output,
        "embedding_model": embedding_model,
        "answer_model": answer_model,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run one baseline1 QA query.")
    parser.add_argument("--question", required=True, help="Input question.")
    parser.add_argument("--docs_path", required=True, help="Path to docs json.")
    parser.add_argument("--vectors_path", required=True, help="Path to vectors npz.")
    parser.add_argument("--answer_model", required=True, help="Answer model used for classification and final answer.")
    parser.add_argument("--embedding_model", required=True, help="Embedding model used for docs/query vectorization.")
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
    parser.add_argument("--embedding_batch_size", type=int, default=64, help="Batch size when building vectors.")
    parser.add_argument("--save_path", default="", help="Optional path to save qa result json.")
    return parser.parse_args()


def resolve_runtime_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    统一解析新旧参数，尽量兼容旧命令。
    """
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
        result = answer_once(
            question=args.question,
            docs_path=args.docs_path,
            vectors_path=args.vectors_path,
            answer_model=args.answer_model,
            embedding_model=args.embedding_model,
            answer_api_key=runtime_args["answer_api_key"],
            answer_base_url=runtime_args["answer_base_url"],
            ark_api_key=runtime_args["ark_api_key"],
            top_k_clips=args.top_k_clips,
            embedding_batch_size=args.embedding_batch_size,
        )

        if args.save_path:
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as file_obj:
                json.dump(result, file_obj, ensure_ascii=False, indent=2)
            LOGGER.info("Saved QA result to %s", save_path)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        LOGGER.exception("qa_once failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
