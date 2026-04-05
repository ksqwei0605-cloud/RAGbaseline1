#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal one-shot RAG QA:
  1) retrieve top-k clips once
  2) concatenate clip memories as context
  3) call OpenAI-compatible chat API once

Usage:
  python qa_once.py --clip-docs clip_docs_with_embeddings.json --question "What happened?" \
    --top-k 3 --backend precomputed --api-key YOUR_KEY --base-url https://api.example.com/v1 \
    --chat-model gpt-4o-mini

  python qa_once.py --clip-docs clip_docs.json --question "What happened?" \
    --top-k 2 --backend tfidf --api-key YOUR_KEY --base-url https://api.example.com/v1 \
    --chat-model gpt-4o-mini

  python qa_once.py --clip-docs clip_docs.json --question "What happened?" \
    --top-k 2 --backend embedding_api --api-key YOUR_KEY --base-url https://api.example.com/v1 \
    --chat-model gpt-4o-mini --embedding-model text-embedding-3-large --save-run run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from retrieval import build_searcher, RetrievalResult


DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant answering questions about a long-video memory graph.\n"
    "Use only the retrieved memory snippets as evidence.\n"
    "Prefer short, direct answers.\n"
    "If the answer is not explicitly stated, make only minimal and reasonable inference.\n"
    "Do not fabricate unsupported details."
)


def build_context(retrieved: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for item in retrieved:
        blocks.append(f"[Clip {item.clip_id}]\n{item.text.strip()}")
    return "\n\n".join(blocks).strip()


def answer_question(
    question: str,
    context: str,
    api_key: str,
    base_url: str | None,
    chat_model: str,
    system_prompt: str,
    temperature: float = 0.2,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package is required for chat API call.") from e

    if not api_key:
        raise ValueError("Missing chat API key (use --api-key or OPENAI_API_KEY).")
    if not chat_model:
        raise ValueError("Missing chat model (use --chat-model or OPENAI_CHAT_MODEL).")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    user_prompt = (
        f"Question:\n{question.strip()}\n\n"
        f"Retrieved memory context:\n{context}\n\n"
        "Answer based only on the context above."
    )

    try:
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Chat API request failed: {e}") from e

    try:
        content = resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to parse chat API response: {e}") from e

    if not content:
        return ""
    return str(content).strip()


def preview_text(text: str, max_chars: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def save_run_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot RAG QA over clip docs.")
    parser.add_argument("--clip-docs", type=str, required=True, help="Path to clip_docs.json (with or without embeddings)")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=2, help="Number of clips to retrieve")
    parser.add_argument(
        "--backend",
        type=str,
        default="precomputed",
        choices=["precomputed", "tfidf", "embedding_api"],
        help="Retrieval backend: precomputed (recommended, needs embeddings), tfidf (fast), embedding_api (API-based)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI-compatible API key (for chat and embedding_api)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--chat-model", type=str, default=None, help="Chat model name (default: OPENAI_CHAT_MODEL or gpt-4o-mini)")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name (for embedding_api backend)")
    parser.add_argument("--embedding-cache", type=str, default=None, help="Optional embedding cache path")
    parser.add_argument("--save-run", type=str, default=None, help="Optional path to save run json")
    parser.add_argument("--system-prompt", type=str, default=None, help="Override default system prompt")
    args = parser.parse_args()

    # Read from environment or use defaults
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    chat_model = args.chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    # Early validation: check critical parameters
    if not api_key:
        print("ERROR: Missing API key. Set --api-key or OPENAI_API_KEY environment variable.")
        return 1
    if not chat_model:
        print("ERROR: Missing chat model. Set --chat-model or OPENAI_CHAT_MODEL environment variable.")
        return 1

    # Build retriever
    try:
        retriever = build_searcher(
            clip_docs_path=args.clip_docs,
            backend=args.backend,
            api_key=api_key,
            base_url=base_url,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize retriever: {e}")
        return 1

    # Retrieve once
    try:
        retrieved = retriever.search_clip(args.question, top_k=args.top_k)
    except Exception as e:
        print(f"ERROR: Retrieval failed: {e}")
        return 1

    if not retrieved:
        print("ERROR: No retrieval results. Please check clip_docs.json and your question.")
        return 1

    context = build_context(retrieved)

    # Ask once
    try:
        final_answer = answer_question(
            question=args.question,
            context=context,
            api_key=api_key,
            base_url=base_url,
            chat_model=chat_model,
            system_prompt=system_prompt,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("=" * 80)
    print("One-shot RAG QA Result")
    print("=" * 80)
    print(f"Question: {args.question}")
    print(f"Retrieval backend: {args.backend}")
    print(f"Top-k: {args.top_k}\n")

    print("Retrieved clips:")
    for i, r in enumerate(retrieved, 1):
        print(f"[{i}] clip_id={r.clip_id} score={r.score:.6f} num_text_nodes={r.num_text_nodes}")
        print(f"    preview: {preview_text(r.text)}")

    print("\nFinal answer:")
    print(final_answer)
    print("=" * 80)

    if args.save_run:
        payload = {
            "question": args.question,
            "retrieval_backend": args.backend,
            "top_k": args.top_k,
            "retrieved_clips": [asdict(x) for x in retrieved],
            "final_answer": final_answer,
            "chat_model": chat_model,
            "base_url": base_url,
        }
        try:
            save_run_json(args.save_run, payload)
            print(f"Saved run json to: {args.save_run}")
        except Exception as e:
            print(f"WARNING: failed to save run json: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
