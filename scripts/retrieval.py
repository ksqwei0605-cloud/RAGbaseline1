#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple clip retrievers for one-shot RAG:
  - tfidf (default, TF-IDF 关键词匹配)
  - precomputed (推荐! 使用预计算的 embeddings，离线快速)
  - embedding_api (使用 OpenAI API 实时嵌入)

Example:
  python retrieval.py --clip-docs clip_docs_with_embeddings.json --question "What happened?" --backend precomputed --top-k 3
  python retrieval.py --clip-docs clip_docs_with_embeddings.json --question "What happened?" --backend tfidf --top-k 2
  python retrieval.py --clip-docs clip_docs.json --question "What happened?" --backend embedding_api \
    --api-key xxx --base-url https://api.example.com/v1 --embedding-model text-embedding-3-large
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class ClipDoc:
    clip_id: Any
    node_ids: list[Any]
    num_text_nodes: int
    text: str
    embeddings: np.ndarray | None = None  # 预计算的向量，可选


@dataclass
class RetrievalResult:
    clip_id: Any
    score: float
    text: str
    node_ids: list[Any]
    num_text_nodes: int


def load_clip_docs(path: str | Path, drop_empty: bool = True) -> list[ClipDoc]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    docs: list[ClipDoc] = []
    for item in raw:
        text = str(item.get("text", "") or "").strip()
        node_ids = item.get("node_ids", [])
        if node_ids is None:
            node_ids = []
        
        # 尝试加载 embeddings（支持两种键名：embeddings 和 embedding）
        embeddings = None
        for key in ["embeddings", "embedding"]:  # 尝试两种键名
            if key in item and item[key] is not None:
                try:
                    embeddings = np.array(item[key], dtype=np.float32)
                    # 假设已经是 (1, dim) 形状，提取为 (dim,)
                    if embeddings.ndim == 2 and embeddings.shape[0] == 1:
                        embeddings = embeddings[0]
                    break  # 找到了就停止
                except Exception:
                    embeddings = None
        
        doc = ClipDoc(
            clip_id=item.get("clip_id"),
            node_ids=list(node_ids),
            num_text_nodes=int(item.get("num_text_nodes", 0)),
            text=text,
            embeddings=embeddings,
        )
        if drop_empty and not text:
            continue
        docs.append(doc)
    return docs


class BaseClipRetriever:
    def search_clip(self, question: str, top_k: int = 2) -> list[RetrievalResult]:
        raise NotImplementedError


class TfidfClipRetriever(BaseClipRetriever):
    def __init__(self, clip_docs: Sequence[ClipDoc]) -> None:
        self.docs = [d for d in clip_docs if d.text.strip()]
        if not self.docs:
            raise ValueError("No non-empty clip docs available for TF-IDF retrieval.")
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in self.docs])

    def search_clip(self, question: str, top_k: int = 2) -> list[RetrievalResult]:
        q = (question or "").strip()
        if not q:
            return []
        q_vec = self.vectorizer.transform([q])
        scores = (self.doc_matrix @ q_vec.T).toarray().ravel()
        if scores.size == 0:
            return []

        k = max(1, min(top_k, len(self.docs)))
        top_idx = np.argsort(-scores)[:k]

        results: list[RetrievalResult] = []
        for idx in top_idx:
            doc = self.docs[int(idx)]
            results.append(
                RetrievalResult(
                    clip_id=doc.clip_id,
                    score=float(scores[int(idx)]),
                    text=doc.text,
                    node_ids=doc.node_ids,
                    num_text_nodes=doc.num_text_nodes,
                )
            )
        return results


class PrecomputedEmbeddingRetriever(BaseClipRetriever):
    """使用预计算的 embeddings 进行检索（离线，无需 API）"""
    
    def __init__(self, clip_docs: Sequence[ClipDoc]) -> None:
        # 过滤出有 embeddings 和非空文本的文档
        self.docs = [d for d in clip_docs if d.text.strip() and d.embeddings is not None]
        if not self.docs:
            raise ValueError("No clip docs with embeddings available for embedding retrieval.")
        
        # 验证所有 embeddings 的维度一致
        embedding_dims = set(d.embeddings.shape[0] for d in self.docs)
        if len(embedding_dims) > 1:
            raise ValueError(f"Inconsistent embedding dimensions: {embedding_dims}")
        
        # 堆叠所有 embeddings 成矩阵 (num_docs, embedding_dim)
        self.doc_embeddings = np.vstack([d.embeddings for d in self.docs])
        self.embedding_dim = self.doc_embeddings.shape[1]
        
        # 预先训练 TF-IDF 向量化器（供问题转换使用）
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            doc_texts = [d.text for d in self.docs]
            self.tfidf_vectorizer = TfidfVectorizer(max_features=min(1000, self.embedding_dim))
            self.tfidf_vectorizer.fit(doc_texts)
        except Exception:
            self.tfidf_vectorizer = None

    def search_clip(self, question: str, top_k: int = 2) -> list[RetrievalResult]:
        """
        使用预计算的 embeddings 检索最相关的 clips
        
        假设所有 embeddings 都已被 L2 归一化，
        所以点积就是余弦相似度。
        """
        q = (question or "").strip()
        if not q:
            return []
        
        # 对问题进行嵌入（需要外部提供或使用简单方法）
        # 这里使用一个简单的 TF-IDF 方法来生成问题向量
        q_embedding = self._embed_question(q)
        if q_embedding is None:
            return []
        
        # 计算与所有文档的相似度（余弦）
        scores = self.doc_embeddings @ q_embedding
        
        k = max(1, min(top_k, len(self.docs)))
        top_idx = np.argsort(-scores)[:k]
        
        results: list[RetrievalResult] = []
        for idx in top_idx:
            doc = self.docs[int(idx)]
            results.append(
                RetrievalResult(
                    clip_id=doc.clip_id,
                    score=float(scores[int(idx)]),
                    text=doc.text,
                    node_ids=doc.node_ids,
                    num_text_nodes=doc.num_text_nodes,
                )
            )
        return results
    
    def _embed_question(self, question: str) -> np.ndarray | None:
        """
        为问题生成 embedding
        
        使用缓存的 TF-IDF 向量化器，将问题转换为与预计算 embeddings 兼容的张量。
        """
        if self.tfidf_vectorizer is None:
            return None
        
        try:
            # 对问题进行 TF-IDF 向量化（使用预训练的 vectorizer）
            q_tfidf = self.tfidf_vectorizer.transform([question]).toarray()[0]
            
            # 如果维度不匹配，需要填充或截断
            if len(q_tfidf) < self.embedding_dim:
                q_tfidf = np.pad(q_tfidf, (0, self.embedding_dim - len(q_tfidf)))
            else:
                q_tfidf = q_tfidf[:self.embedding_dim]
            
            # 归一化
            norm = np.linalg.norm(q_tfidf)
            if norm > 0:
                q_tfidf = q_tfidf / norm
            
            return q_tfidf.astype(np.float32)
        except Exception:
            return None


class EmbeddingAPIClipRetriever(BaseClipRetriever):
    def __init__(
        self,
        clip_docs: Sequence[ClipDoc],
        api_key: str,
        base_url: str | None,
        embedding_model: str,
        cache_path: str | Path | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for embedding_api backend.")
        if not embedding_model:
            raise ValueError("embedding_model is required for embedding_api backend.")

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package is required for embedding_api backend.") from e

        self.docs = [d for d in clip_docs if d.text.strip()]
        if not self.docs:
            raise ValueError("No non-empty clip docs available for embedding retrieval.")

        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.cache_path = Path(cache_path) if cache_path else None

        self.doc_embeddings = self._load_or_build_doc_embeddings()

    def _docs_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.embedding_model.encode("utf-8"))
        for d in self.docs:
            h.update(str(d.clip_id).encode("utf-8"))
            h.update(b"\n")
            h.update(d.text.encode("utf-8"))
            h.update(b"\n---\n")
        return h.hexdigest()

    def _embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        all_vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.embedding_model, input=batch)
            vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
            all_vectors.extend(vecs)

        mat = np.vstack(all_vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = mat / norms
        return mat

    def _load_or_build_doc_embeddings(self) -> np.ndarray:
        fingerprint = self._docs_fingerprint()
        if self.cache_path and self.cache_path.exists():
            try:
                with self.cache_path.open("rb") as f:
                    payload = pickle.load(f)
                if (
                    isinstance(payload, dict)
                    and payload.get("fingerprint") == fingerprint
                    and payload.get("model") == self.embedding_model
                    and isinstance(payload.get("embeddings"), np.ndarray)
                ):
                    emb = payload["embeddings"]
                    if emb.shape[0] == len(self.docs):
                        return emb.astype(np.float32)
            except Exception:
                pass

        texts = [d.text for d in self.docs]
        emb = self._embed_texts(texts)

        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("wb") as f:
                    pickle.dump(
                        {
                            "fingerprint": fingerprint,
                            "model": self.embedding_model,
                            "embeddings": emb,
                        },
                        f,
                    )
            except Exception:
                # Cache failure should not block retrieval.
                pass

        return emb

    def search_clip(self, question: str, top_k: int = 2) -> list[RetrievalResult]:
        q = (question or "").strip()
        if not q:
            return []

        q_emb = self._embed_texts([q])[0]  # normalized
        scores = self.doc_embeddings @ q_emb

        k = max(1, min(top_k, len(self.docs)))
        top_idx = np.argsort(-scores)[:k]

        results: list[RetrievalResult] = []
        for idx in top_idx:
            doc = self.docs[int(idx)]
            results.append(
                RetrievalResult(
                    clip_id=doc.clip_id,
                    score=float(scores[int(idx)]),
                    text=doc.text,
                    node_ids=doc.node_ids,
                    num_text_nodes=doc.num_text_nodes,
                )
            )
        return results


def build_searcher(
    clip_docs_path: str | Path,
    backend: str = "precomputed",  # 默认改为 precomputed（更优）
    api_key: str | None = None,
    base_url: str | None = None,
    embedding_model: str | None = None,
    embedding_cache: str | Path | None = None,
) -> BaseClipRetriever:
    docs = load_clip_docs(clip_docs_path, drop_empty=True)
    if not docs:
        raise ValueError("No valid clip docs loaded.")

    backend = backend.strip().lower()
    
    # 优先级1：预计算的 embeddings（最高效）
    if backend == "precomputed":
        # 检查是否有文档带有 embeddings
        docs_with_emb = [d for d in docs if d.embeddings is not None]
        if not docs_with_emb:
            raise ValueError(
                "No embeddings found in clip docs. "
                "Please use clip_docs_with_embeddings.json or generate embeddings first."
            )
        return PrecomputedEmbeddingRetriever(docs)
    
    # 优先级2：TF-IDF（快速备选）
    if backend == "tfidf":
        return TfidfClipRetriever(docs)

    # 优先级3：OpenAI API（精准但需要网络）
    if backend == "embedding_api":
        final_api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        final_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        final_embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "")
        if not final_api_key:
            raise ValueError("Missing API key for embedding_api (use --api-key or OPENAI_API_KEY).")
        if not final_embedding_model:
            raise ValueError(
                "Missing embedding model for embedding_api (use --embedding-model or OPENAI_EMBEDDING_MODEL)."
            )

        if embedding_cache is None:
            p = Path(clip_docs_path)
            model_tag = hashlib.md5(final_embedding_model.encode("utf-8")).hexdigest()[:8]
            embedding_cache = p.with_suffix(f".embcache.{model_tag}.pkl")

        return EmbeddingAPIClipRetriever(
            clip_docs=docs,
            api_key=final_api_key,
            base_url=final_base_url,
            embedding_model=final_embedding_model,
            cache_path=embedding_cache,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Simple clip retrieval CLI.")
    parser.add_argument("--clip-docs", type=str, required=True, help="Path to clip_docs JSON (with or without embeddings)")
    parser.add_argument("--question", type=str, required=True, help="Question to search")
    parser.add_argument(
        "--backend",
        type=str,
        default="precomputed",
        choices=["precomputed", "tfidf", "embedding_api"],
        help="Retrieval backend: precomputed (rec, embeddings), tfidf (fast), embedding_api (API-based)",
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (for embedding_api backend)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI base URL (for embedding_api backend)")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name (for embedding_api backend)")
    parser.add_argument("--embedding-cache", type=str, default=None, help="Embedding cache file path (for embedding_api backend)")
    args = parser.parse_args()

    try:
        retriever = build_searcher(
            clip_docs_path=args.clip_docs,
            backend=args.backend,
            api_key=args.api_key,
            base_url=args.base_url,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
        )
        results = retriever.search_clip(args.question, top_k=args.top_k)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("=" * 80)
    print(f"Question: {args.question}")
    print(f"Backend: {args.backend}")
    print("=" * 80)
    
    if not results:
        print("No results found.")
    else:
        for i, r in enumerate(results, 1):
            preview = " ".join(r.text.split())
            if len(preview) > 220:
                preview = preview[:217] + "..."
            print(f"[{i}] clip_id={r.clip_id} score={r.score:.6f} num_text_nodes={r.num_text_nodes}")
            print(f"    {preview}")

    print("\nRaw JSON output:")
    print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    print(f"\nTotal results: {len(results)} / top_k: {args.top_k}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
