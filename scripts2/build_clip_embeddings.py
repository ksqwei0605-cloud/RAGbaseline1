#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个文件的主要功能：
读取 build_clip_docs.py 生成的 docs.json，使用外部 embedding_model
为每个 node 的文本生成向量，并保存为 vectors.npz。

说明：
- 检索统一使用外部 embedding_model 生成的向量。
- 不使用 pkl 中原始 node.embeddings。
- 输出 npz 结构固定为 embeddings / node_ids / clip_ids / type_ids。
- 当前 embedding 调用基于火山方舟 Ark SDK。
- 这里 embedding 优先使用“角色归一化文本” normalized_content_text，
  这样 face/voice 到 character 的统一信息能反映到向量空间里。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("build_clip_embeddings")
TYPE_TO_ID = {"episodic": 0, "semantic": 1}


def setup_logging() -> None:
    """初始化日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def load_docs_json(docs_path: str) -> Dict[str, Any]:
    """
    读取 docs.json。

    输入：
    - docs_path: docs json 路径

    输出：
    - json 对应的字典
    """
    path = Path(docs_path)
    if not path.is_file():
        raise FileNotFoundError("Docs json not found: {0}".format(path))

    LOGGER.info("Loading docs json from %s", path)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def type_to_id(node_type: str) -> int:
    """
    将节点类型编码为固定整数。

    输入：
    - node_type: episodic 或 semantic

    输出：
    - 0 或 1
    """
    if node_type not in TYPE_TO_ID:
        raise ValueError("Unsupported node type: {0}".format(node_type))
    return TYPE_TO_ID[node_type]


def choose_embedding_text(node_item: Dict[str, Any]) -> str:
    """
    选择用于 embedding 的文本。

    规则：
    - 优先使用 normalized_content_text
    - 若为空则回退到 content_text
    """
    normalized_content_text = str(node_item.get("normalized_content_text", "")).strip()
    if normalized_content_text:
        return normalized_content_text
    return str(node_item.get("content_text", "")).strip()


def collect_node_texts(docs_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 docs 数据中收集 embedding 所需的节点文本与元信息。

    输出的每条记录至少包含：
    - clip_id
    - node_id
    - type
    - content_text
    - normalized_content_text
    - embedding_text
    """
    clips = docs_data.get("clips", [])
    if not isinstance(clips, list):
        raise ValueError("docs_data['clips'] must be a list.")

    records: List[Dict[str, Any]] = []
    for clip_item in clips:
        clip_id = clip_item.get("clip_id")
        nodes = clip_item.get("nodes", [])
        if not isinstance(nodes, list):
            LOGGER.warning("clip_id=%s has invalid nodes structure, skip.", clip_id)
            continue

        for node_item in nodes:
            content_text = str(node_item.get("content_text", "")).strip()
            normalized_content_text = str(node_item.get("normalized_content_text", "")).strip()
            embedding_text = choose_embedding_text(node_item)
            if not embedding_text:
                LOGGER.warning(
                    "clip_id=%s node_id=%s has empty embedding text, skip.",
                    clip_id,
                    node_item.get("node_id"),
                )
                continue

            records.append(
                {
                    "clip_id": int(clip_id),
                    "node_id": int(node_item["node_id"]),
                    "type": str(node_item["type"]).strip(),
                    "content_text": content_text,
                    "normalized_content_text": normalized_content_text,
                    "embedding_text": embedding_text,
                }
            )

    return records


def batch_iter(items: List[str], batch_size: int) -> List[List[str]]:
    """
    将文本列表切成批次。

    输入：
    - items: 文本列表
    - batch_size: 批大小

    输出：
    - 批次列表
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def create_ark_client(ark_api_key: str) -> Any:
    """
    创建火山方舟 Ark 客户端。

    输入：
    - ark_api_key: Ark API key

    输出：
    - Ark 客户端
    """
    try:
        from volcenginesdkarkruntime import Ark
    except ImportError as exc:
        raise ImportError(
            "volcenginesdkarkruntime package is required for Ark embedding requests. "
            "Please install it first, for example: pip install volcengine-python-sdk[ark]"
        ) from exc

    if not ark_api_key:
        raise ValueError("Missing Ark API key for embedding requests.")

    return Ark(api_key=ark_api_key)


def extract_embedding_vector_from_response(response: Any) -> List[float]:
    """
    从 Ark embedding 响应中稳健提取向量。

    兼容的常见情况：
    - response.data 是单个对象，且有 embedding 属性
    - response.data 是列表，列表元素有 embedding 属性
    - response.data 是字典，含 embedding 字段
    """
    data = getattr(response, "data", None)
    if data is None:
        raise RuntimeError("Ark embedding API response does not contain data.")

    candidate = data
    if isinstance(data, list):
        if not data:
            raise RuntimeError("Ark embedding API response.data is empty.")
        candidate = data[0]

    if hasattr(candidate, "embedding"):
        vector = getattr(candidate, "embedding")
    elif isinstance(candidate, dict) and "embedding" in candidate:
        vector = candidate["embedding"]
    else:
        raise RuntimeError(
            "Unsupported Ark embedding response structure: data_type={0}".format(type(candidate).__name__)
        )

    vector_list = list(vector)
    if not vector_list:
        raise RuntimeError("Ark embedding API returned an empty embedding vector.")
    return vector_list


def get_single_embedding_from_ark(client: Any, text: str, embedding_model: str) -> List[float]:
    """
    调用 Ark multimodal embedding API，为单条文本生成向量。

    输入：
    - client: Ark 客户端
    - text: 输入文本
    - embedding_model: embedding 模型名

    输出：
    - 单条 embedding 向量
    """
    response = client.multimodal_embeddings.create(
        model=embedding_model,
        input=[
            {
                "type": "text",
                "text": text,
            }
        ],
    )

    if not hasattr(response, "data") or not response.data:
        raise RuntimeError("Ark embedding API returned empty result.")

    return extract_embedding_vector_from_response(response)


def get_embeddings_from_api(
    texts: List[str],
    embedding_model: str,
    ark_api_key: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    调用 Ark multimodal embedding API，批量生成文本向量。

    输入：
    - texts: 文本列表
    - embedding_model: embedding 模型名
    - ark_api_key: Ark API key
    - batch_size: 每批文本数，仅用于控制日志和处理节奏

    输出：
    - shape=(num_texts, dim) 的 float32 numpy 数组
    """
    if not texts:
        raise ValueError("texts is empty.")
    if not embedding_model:
        raise ValueError("embedding_model is required.")

    client = create_ark_client(ark_api_key=ark_api_key)
    all_embeddings: List[List[float]] = []
    expected_dim = None
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_index, text_batch in enumerate(batch_iter(texts, batch_size), start=1):
        LOGGER.info(
            "Requesting Ark embeddings batch %s/%s with %s texts.",
            batch_index,
            total_batches,
            len(text_batch),
        )

        batch_embeddings: List[List[float]] = []
        for item_index, text in enumerate(text_batch, start=1):
            vector = get_single_embedding_from_ark(client=client, text=text, embedding_model=embedding_model)
            if expected_dim is None:
                expected_dim = len(vector)
            elif len(vector) != expected_dim:
                raise ValueError(
                    "Embedding dimension mismatch. expected_dim={0}, got={1}".format(expected_dim, len(vector))
                )
            batch_embeddings.append(vector)

            if item_index % 10 == 0 or item_index == len(text_batch):
                LOGGER.info(
                    "Batch %s/%s progress: %s/%s texts embedded.",
                    batch_index,
                    total_batches,
                    item_index,
                    len(text_batch),
                )

        all_embeddings.extend(batch_embeddings)
        LOGGER.info("Finished batch %s/%s.", batch_index, total_batches)

    embeddings = np.asarray(all_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings array should be 2D, got shape={0}".format(embeddings.shape))
    return embeddings


def save_embeddings_npz(
    out_path: str,
    embeddings: np.ndarray,
    node_ids: np.ndarray,
    clip_ids: np.ndarray,
    type_ids: np.ndarray,
) -> None:
    """
    保存 vectors.npz。

    输入：
    - out_path: 输出 npz 路径
    - embeddings: (num_nodes, dim)
    - node_ids: (num_nodes,)
    - clip_ids: (num_nodes,)
    - type_ids: (num_nodes,)
    """
    path = Path(out_path)
    ensure_dir(path.parent)
    np.savez(
        path,
        embeddings=embeddings,
        node_ids=node_ids,
        clip_ids=clip_ids,
        type_ids=type_ids,
    )
    LOGGER.info("Saved vectors npz to %s", path)


def save_meta_json(out_path: str, meta_data: Dict[str, Any]) -> None:
    """
    可选保存一个简洁 meta json，便于调试。
    """
    path = Path(out_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(meta_data, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("Saved vectors meta json to %s", path)


def build_clip_embeddings(
    docs_path: str,
    output_dir: str,
    embedding_model: str,
    ark_api_key: str,
    batch_size: int = 64,
) -> str:
    """
    主流程：从 docs.json 构建 vectors.npz。

    输入：
    - docs_path: docs.json 路径
    - output_dir: 输出目录
    - embedding_model: embedding 模型名
    - ark_api_key: Ark API key

    输出：
    - 生成的 vectors.npz 路径
    """
    docs_data = load_docs_json(docs_path)
    records = collect_node_texts(docs_data)
    if not records:
        raise ValueError("No valid node texts found in docs json.")

    pkl_name = str(docs_data.get("pkl_name", Path(docs_path).stem.replace("_docs", "")))
    LOGGER.info("Collected %s nodes for embedding generation.", len(records))

    texts = [item["embedding_text"] for item in records]
    embeddings = get_embeddings_from_api(
        texts=texts,
        embedding_model=embedding_model,
        ark_api_key=ark_api_key,
        batch_size=batch_size,
    )

    node_ids = np.asarray([item["node_id"] for item in records], dtype=np.int64)
    clip_ids = np.asarray([item["clip_id"] for item in records], dtype=np.int64)
    type_ids = np.asarray([type_to_id(item["type"]) for item in records], dtype=np.int64)

    if not (len(embeddings) == len(node_ids) == len(clip_ids) == len(type_ids)):
        raise ValueError("embeddings, node_ids, clip_ids, type_ids length mismatch.")

    output_dir_path = Path(output_dir)
    vectors_path = output_dir_path / "{0}_vectors.npz".format(pkl_name)
    meta_path = output_dir_path / "{0}_vectors_meta.json".format(pkl_name)

    save_embeddings_npz(
        out_path=str(vectors_path),
        embeddings=embeddings,
        node_ids=node_ids,
        clip_ids=clip_ids,
        type_ids=type_ids,
    )
    save_meta_json(
        out_path=str(meta_path),
        meta_data={
            "pkl_name": pkl_name,
            "embedding_model": embedding_model,
            "num_nodes": int(len(records)),
            "embedding_dim": int(embeddings.shape[1]),
            "vectors_path": str(vectors_path),
            "num_normalized_texts": int(
                sum(
                    1
                    for item in records
                    if item["normalized_content_text"] and item["normalized_content_text"] != item["content_text"]
                )
            ),
        },
    )

    LOGGER.info("Successfully generated %s node embeddings.", len(records))
    LOGGER.info("Embedding dimension: %s", embeddings.shape[1])
    LOGGER.info("Output vectors path: %s", vectors_path)
    return str(vectors_path)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build baseline1 node embeddings from docs json.")
    parser.add_argument("--docs_path", required=True, help="Path to docs json generated by build_clip_docs.py.")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "baseline1" / "outputs"),
        help="Directory to save outputs/{pkl_name}_vectors.npz.",
    )
    parser.add_argument("--embedding_model", required=True, help="Embedding model name.")
    parser.add_argument(
        "--ark_api_key",
        default=os.getenv("ARK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        help="Ark API key for embedding requests.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding API.")
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""
    setup_logging()
    args = parse_args()

    try:
        build_clip_embeddings(
            docs_path=args.docs_path,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            ark_api_key=args.ark_api_key,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        LOGGER.exception("Failed to build embeddings: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
