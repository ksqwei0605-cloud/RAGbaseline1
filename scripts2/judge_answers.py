#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个脚本的主要功能：
读取 run_baseline1.py 生成的 baseline 结果 json，
对每道题的 question / gold_answer / pred_answer 调用大模型做自动裁判，
输出逐题判定结果、整体 accuracy，并保存 judged 结果文件与 summary 文件。

说明：
- 主裁判模型默认使用 Qwen3-Max。
- 可选复核模型默认使用 GLM。
- API 调用采用 OpenAI-compatible 接口风格，方便后续替换。
- 默认只使用主裁判；当开启复核时，可按规则触发 GLM 复核。
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger("judge_answers")
UNCERTAIN_HINTS = ("uncertain", "ambiguous", "insufficient", "unclear")
DEFAULT_MAIN_MODEL = "qwen3-max"
DEFAULT_REVIEW_MODEL = "glm-4.5"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
VALID_LABELS = {"correct", "incorrect"}
VALID_MERGE_STRATEGIES = {"review_first", "main_first", "majority"}


def setup_logging() -> None:
    """初始化日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def str2bool(value: Any) -> bool:
    """将常见字符串解析为 bool。"""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got: {0}".format(value))


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def load_results_json(input_path: str) -> Dict[str, Any]:
    """
    读取 baseline 结果 json。

    要求：
    - 顶层是 dict
    - 包含 results 列表
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError("Input results json not found: {0}".format(path))

    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise ValueError("Input results json top-level must be a dict.")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Input results json must contain a list field named 'results'.")

    LOGGER.info("Loaded %s QA items from %s", len(results), path)
    return data


def build_judge_prompt(question: str, gold_answer: str, pred_answer: str) -> List[Dict[str, str]]:
    """
    构造裁判 prompt。

    要求模型判断：
    - pred_answer 是否与 gold_answer 在核心语义上等价或足够一致
    - 并只输出严格 JSON
    """
    system_prompt = (
        "You are an expert judge for question answering evaluation.\n"
        "Your task is to decide whether the predicted answer is semantically correct with respect to the gold answer.\n"
        "Focus on meaning, factual correctness, and whether the key conclusion is preserved.\n"
        "Do not judge by wording alone.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "judge_label": "correct" or "incorrect",\n'
        '  "correct": true or false,\n'
        '  "reason": "brief explanation"\n'
        "}\n"
        "Do not output markdown, code fences, or extra commentary."
    )

    user_prompt = (
        "Task:\n"
        "Given a question, a gold answer, and a predicted answer, judge whether the predicted answer is correct.\n\n"
        "Judging standard:\n"
        "- Mark correct if the predicted answer preserves the core conclusion of the gold answer.\n"
        "- Different wording, paraphrase, or a longer but semantically consistent answer should still be correct.\n"
        "- Mark incorrect if the prediction misses a key conclusion, states the opposite, gives wrong facts, "
        "is clearly incomplete in a way that changes correctness, or refuses to answer while the gold answer is answerable.\n"
        "- A very conservative answer like 'uncertain' or 'cannot determine' is usually incorrect when the gold answer is explicit.\n"
        "- If extra details are present but do not change the core meaning, it can still be correct.\n"
        "- If extra details alter the core conclusion, it should be incorrect.\n\n"
        "Few-shot examples:\n"
        "Example 1:\n"
        'question: "When does Lily need afternoon tea originally?"\n'
        'gold_answer: "About 4pm"\n'
        'pred_answer: "At around ten to four."\n'
        'output: {"judge_label":"correct","correct":true,"reason":"The time expression is semantically equivalent to about 4pm."}\n\n'
        "Example 2:\n"
        'question: "What is Lily\'s occupation?"\n'
        'gold_answer: "Designer"\n'
        'pred_answer: "The answer is uncertain."\n'
        'output: {"judge_label":"incorrect","correct":false,"reason":"The prediction refuses to answer while the gold answer is explicit."}\n\n'
        "Example 3:\n"
        'question: "How does Emma plan to improve her programming skills?"\n'
        'gold_answer: "By joining new projects, working on innovative applications, and participating in programming competitions."\n'
        'pred_answer: "Emma plans to improve her programming skills by joining new projects, working on innovative applications, participating in programming competitions, and she has already started small-scale development work."\n'
        'output: {"judge_label":"correct","correct":true,"reason":"The key plan matches the gold answer and the extra detail does not change the core conclusion."}\n\n'
        "Example 4:\n"
        'question: "What is Emma\'s afternoon tea?"\n'
        'gold_answer: "French fries, ketchup and banana cake"\n'
        'pred_answer: "French fries"\n'
        'output: {"judge_label":"incorrect","correct":false,"reason":"The prediction omits key items from the gold answer."}\n\n'
        "Example 5:\n"
        'question: "Is Emma\'s tidying up habit good?"\n'
        'gold_answer: "Not good"\n'
        'pred_answer: "Yes, it is good."\n'
        'output: {"judge_label":"incorrect","correct":false,"reason":"The prediction states the opposite conclusion."}\n\n'
        "Now judge this case.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Gold answer:\n{gold_answer.strip()}\n\n"
        f"Predicted answer:\n{pred_answer.strip()}\n\n"
        "Output JSON only."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def create_openai_client(api_key: str, base_url: str = "") -> Any:
    """创建 OpenAI-compatible 客户端。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required. Please install it with: pip install openai") from exc

    if not api_key:
        raise ValueError("Missing API key for model requests.")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def call_chat_model(
    messages: List[Dict[str, str]],
    model: str,
    base_url: str,
    api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
    temperature: float = 0.0,
) -> str:
    """
    调用 OpenAI-compatible chat model。

    特性：
    - 支持重试
    - 打印关键错误日志
    - 失败时抛出异常，由上层决定是否继续
    """
    client = create_openai_client(api_key=api_key, base_url=base_url)
    attempts = max(int(max_retries), 1)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            content = response.choices[0].message.content
            return str(content).strip() if content else ""
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Model call failed on attempt %s/%s for model=%s: %s",
                attempt,
                attempts,
                model,
                exc,
            )
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 5))

    raise RuntimeError("Model call failed after {0} retries: {1}".format(attempts, last_error))


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型返回文本中尽量提取 JSON 对象。

    支持：
    - 纯 JSON
    - ```json fenced block
    - 文本中嵌入一个 JSON 对象
    """
    raw_text = str(text or "").strip()
    if not raw_text:
        return None

    candidates: List[str] = []

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    if raw_text.startswith("{") and raw_text.endswith("}"):
        candidates.append(raw_text)

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(raw_text[first_brace:last_brace + 1].strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed_obj, end_index = decoder.raw_decode(raw_text[index:])
            if isinstance(parsed_obj, dict):
                return parsed_obj
        except Exception:
            continue

    return None


def normalize_label_and_bool(judge_label: str, correct: Any) -> Tuple[Optional[str], Optional[bool]]:
    """标准化 judge_label 和 correct。"""
    label = str(judge_label or "").strip().lower()
    if label not in VALID_LABELS:
        label = ""

    normalized_correct: Optional[bool]
    if isinstance(correct, bool):
        normalized_correct = correct
    elif str(correct).strip().lower() in {"true", "1", "yes"}:
        normalized_correct = True
    elif str(correct).strip().lower() in {"false", "0", "no"}:
        normalized_correct = False
    else:
        normalized_correct = None

    if label and normalized_correct is None:
        normalized_correct = label == "correct"
    if normalized_correct is not None and not label:
        label = "correct" if normalized_correct else "incorrect"

    if not label or normalized_correct is None:
        return None, None
    return label, normalized_correct


def parse_judge_result(raw_text: str) -> Dict[str, Any]:
    """
    解析裁判模型输出。

    返回结构包含：
    - parsed_ok
    - judge_label
    - correct
    - reason
    - parse_error
    """
    parsed = extract_json_from_response(raw_text)
    if not isinstance(parsed, dict):
        return {
            "parsed_ok": False,
            "judge_label": "incorrect",
            "correct": False,
            "reason": "Judge response JSON parsing failed.",
            "parse_error": "No valid JSON object found in judge response.",
        }

    judge_label, correct = normalize_label_and_bool(
        judge_label=parsed.get("judge_label"),
        correct=parsed.get("correct"),
    )
    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided by judge model."

    if judge_label is None or correct is None:
        return {
            "parsed_ok": False,
            "judge_label": "incorrect",
            "correct": False,
            "reason": reason,
            "parse_error": "Missing or invalid judge_label/correct in parsed JSON.",
        }

    return {
        "parsed_ok": True,
        "judge_label": judge_label,
        "correct": correct,
        "reason": reason,
        "parse_error": "",
    }


def judge_with_model(
    question: str,
    gold_answer: str,
    pred_answer: str,
    model: str,
    base_url: str,
    api_key: str,
    max_retries: int,
    timeout: float,
) -> Dict[str, Any]:
    """
    使用指定模型做裁判，并返回原始输出与解析结果。
    """
    messages = build_judge_prompt(question, gold_answer, pred_answer)

    try:
        raw_text = call_chat_model(
            messages=messages,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            temperature=0.0,
        )
        parsed_result = parse_judge_result(raw_text)
        parsed_result["raw_text"] = raw_text
        parsed_result["judge_model"] = model
        return parsed_result
    except Exception as exc:
        return {
            "parsed_ok": False,
            "judge_label": "incorrect",
            "correct": False,
            "reason": "Judge model request failed.",
            "parse_error": str(exc),
            "raw_text": "",
            "judge_model": model,
        }


def judge_with_main_model(
    question: str,
    gold_answer: str,
    pred_answer: str,
    model: str,
    base_url: str,
    api_key: str,
    max_retries: int,
    timeout: float,
) -> Dict[str, Any]:
    """使用主裁判模型裁判。"""
    return judge_with_model(
        question=question,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        timeout=timeout,
    )


def should_trigger_review(main_result: Dict[str, Any], raw_text: str, review_all: bool = False) -> bool:
    """
    判断是否触发复核。

    触发条件：
    - review_all=True
    - 主裁判输出无法解析
    - raw_text 或 reason 中包含 uncertain / ambiguous / insufficient / unclear
    """
    if review_all:
        return True

    if not bool(main_result.get("parsed_ok")):
        return True

    merged_text = "{0}\n{1}".format(str(raw_text or ""), str(main_result.get("reason", ""))).lower()
    return any(hint in merged_text for hint in UNCERTAIN_HINTS)


def judge_with_review_model(
    question: str,
    gold_answer: str,
    pred_answer: str,
    model: str,
    base_url: str,
    api_key: str,
    max_retries: int,
    timeout: float,
) -> Dict[str, Any]:
    """使用复核模型裁判。"""
    return judge_with_model(
        question=question,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        timeout=timeout,
    )


def choose_majority_with_tiebreak(
    main_result: Dict[str, Any],
    review_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    majority 策略的实现。

    说明：
    - 目前只有两个裁判，因此当二者不一致时不存在真正多数。
    - 这里默认用 review 作为平局时的 tie-break，便于后续扩展到第三个裁判。
    """
    if main_result.get("judge_label") == review_result.get("judge_label"):
        return review_result
    return review_result


def merge_judge_results(
    main_result: Dict[str, Any],
    review_result: Optional[Dict[str, Any]],
    strategy: str = "review_first",
) -> Dict[str, Any]:
    """
    合并主裁判与复核裁判结果。

    默认：
    - 一致则采用一致结果
    - 不一致则按策略合并
    - 若只有一个可解析，则优先用可解析结果
    """
    if strategy not in VALID_MERGE_STRATEGIES:
        raise ValueError("Unsupported merge strategy: {0}".format(strategy))

    if review_result is None:
        return dict(main_result)

    main_ok = bool(main_result.get("parsed_ok"))
    review_ok = bool(review_result.get("parsed_ok"))

    if review_ok and not main_ok:
        return dict(review_result)
    if main_ok and not review_ok:
        return dict(main_result)
    if not main_ok and not review_ok:
        merged = dict(review_result if strategy != "main_first" else main_result)
        merged["judge_label"] = "incorrect"
        merged["correct"] = False
        merged["reason"] = (
            "Both main judge and review judge failed to return a reliably parseable result. "
            "Defaulted to incorrect."
        )
        merged["parsed_ok"] = False
        return merged

    if main_result.get("judge_label") == review_result.get("judge_label"):
        return dict(review_result if strategy != "main_first" else main_result)

    if strategy == "main_first":
        return dict(main_result)
    if strategy == "review_first":
        return dict(review_result)
    return choose_majority_with_tiebreak(main_result, review_result)


def build_compact_result_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    构造精简版 judged item。

    说明：
    - 默认不保留 retrieved_clips / retrieved_nodes / context 这类大字段
    - 仅保留判题与复盘最关键的信息
    """
    return {
        "question_id": item.get("question_id", ""),
        "pkl_name": item.get("pkl_name", ""),
        "question": item.get("question", ""),
        "gold_answer": item.get("gold_answer", ""),
        "pred_answer": item.get("pred_answer", ""),
        "field": item.get("field", ""),
        "classifier_raw_output": item.get("classifier_raw_output", ""),
        "embedding_model": item.get("embedding_model", ""),
        "answer_model": item.get("answer_model", ""),
        "character_hints": item.get("character_hints", []),
    }


def judge_single_item(
    item: Dict[str, Any],
    main_model: str,
    main_base_url: str,
    main_api_key: str,
    review_model: str,
    review_base_url: str,
    review_api_key: str,
    enable_review: bool,
    review_all: bool,
    merge_strategy: str,
    max_retries: int,
    timeout: float,
) -> Dict[str, Any]:
    """
    对单条样本进行裁判。

    特点：
    - 尽量不中断全流程
    - 默认输出精简字段，避免 judged 文件过长
    """
    result_item = build_compact_result_item(item)
    question = str(item.get("question", "")).strip()
    gold_answer = str(item.get("gold_answer", "")).strip()
    pred_answer = str(item.get("pred_answer", "")).strip()

    main_result = judge_with_main_model(
        question=question,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        model=main_model,
        base_url=main_base_url,
        api_key=main_api_key,
        max_retries=max_retries,
        timeout=timeout,
    )

    review_result: Optional[Dict[str, Any]] = None
    used_review = False
    review_triggered = False

    if enable_review:
        review_triggered = should_trigger_review(
            main_result=main_result,
            raw_text=str(main_result.get("raw_text", "")),
            review_all=review_all,
        )
        if review_triggered:
            used_review = True
            LOGGER.info(
                "Review triggered for question_id=%s with main label=%s parsed_ok=%s",
                item.get("question_id", ""),
                main_result.get("judge_label"),
                main_result.get("parsed_ok"),
            )
            review_result = judge_with_review_model(
                question=question,
                gold_answer=gold_answer,
                pred_answer=pred_answer,
                model=review_model,
                base_url=review_base_url,
                api_key=review_api_key,
                max_retries=max_retries,
                timeout=timeout,
            )

    final_result = merge_judge_results(
        main_result=main_result,
        review_result=review_result,
        strategy=merge_strategy,
    )

    result_item.update(
        {
            "judge_model": final_result.get("judge_model", main_model),
            "correct": bool(final_result.get("correct", False)),
            "judge_label": str(final_result.get("judge_label", "incorrect")),
            "reason": str(final_result.get("reason", "")),
            "main_judge_raw": str(main_result.get("raw_text", "")),
            "review_judge_raw": str(review_result.get("raw_text", "")) if review_result else "",
            "final_judge_model": str(final_result.get("judge_model", main_model)),
            "used_review": used_review,
            "review_triggered": review_triggered,
            "main_judge_parsed_ok": bool(main_result.get("parsed_ok", False)),
            "review_judge_parsed_ok": bool(review_result.get("parsed_ok", False)) if review_result else False,
            "main_judge_parse_error": str(main_result.get("parse_error", "")),
            "review_judge_parse_error": str(review_result.get("parse_error", "")) if review_result else "",
        }
    )
    return result_item


def compute_accuracy(judged_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算整体 accuracy。

    定义：
    - correct 题数 / 总题数
    """
    total = len(judged_items)
    num_correct = sum(1 for item in judged_items if bool(item.get("correct")) is True)
    num_incorrect = total - num_correct
    accuracy = (num_correct / total) if total > 0 else 0.0
    return {
        "total": total,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "accuracy": accuracy,
    }


def save_judged_results(output_path: str, data: Dict[str, Any]) -> None:
    """保存详细 judged 结果文件。"""
    path = Path(output_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("Saved judged results to %s", path)


def save_summary(output_path: str, data: Dict[str, Any]) -> None:
    """保存 summary 文件。"""
    path = Path(output_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("Saved judge summary to %s", path)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Judge baseline1 QA results with Qwen and optional GLM review.")

    parser.add_argument("--input_path", required=True, help="Path to baseline result json generated by run_baseline1.py.")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory to save judged json files. Default: use the input file's parent directory.",
    )

    parser.add_argument(
        "--main_model",
        default=DEFAULT_MAIN_MODEL,
        help="Main judge model name. Default: {0}".format(DEFAULT_MAIN_MODEL),
    )
    parser.add_argument(
        "--main_base_url",
        default="",
        help="Base URL for main judge model. If empty, will try QWEN_BASE_URL then OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--main_api_key",
        default="",
        help="API key for main judge model. If empty, will try QWEN_API_KEY then OPENAI_API_KEY.",
    )

    parser.add_argument(
        "--review_model",
        default=DEFAULT_REVIEW_MODEL,
        help="Review judge model name. Default: {0}".format(DEFAULT_REVIEW_MODEL),
    )
    parser.add_argument(
        "--review_base_url",
        default="",
        help="Base URL for review judge model. If empty, will try GLM_BASE_URL then OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--review_api_key",
        default="",
        help="API key for review judge model. If empty, will try GLM_API_KEY then OPENAI_API_KEY.",
    )

    parser.add_argument(
        "--enable_review",
        type=str2bool,
        default=False,
        help="Whether to enable review model. Default: false.",
    )
    parser.add_argument(
        "--review_all",
        type=str2bool,
        default=False,
        help="Whether to review all samples regardless of main judge confidence. Default: false.",
    )
    parser.add_argument(
        "--merge_strategy",
        default="review_first",
        choices=sorted(VALID_MERGE_STRATEGIES),
        help="How to merge main and review judge results. Default: review_first.",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for each model API call. Default: {0}".format(DEFAULT_MAX_RETRIES),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for each API call. Default: {0}".format(DEFAULT_TIMEOUT),
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    统一解析 CLI 参数与环境变量。

    优先级：
    - CLI 参数
    - 专属环境变量
    - OPENAI_* 通用环境变量
    """
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    main_api_key = args.main_api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    main_base_url = args.main_base_url or os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")

    review_api_key = args.review_api_key or os.getenv("GLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    review_base_url = args.review_base_url or os.getenv("GLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")

    if not main_api_key:
        raise ValueError("Main judge API key is missing. Provide --main_api_key or set QWEN_API_KEY / OPENAI_API_KEY.")

    if bool(args.enable_review) and not review_api_key:
        raise ValueError(
            "Review judge API key is missing while review is enabled. "
            "Provide --review_api_key or set GLM_API_KEY / OPENAI_API_KEY."
        )

    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "main_model": args.main_model,
        "main_base_url": main_base_url,
        "main_api_key": main_api_key,
        "review_model": args.review_model,
        "review_base_url": review_base_url,
        "review_api_key": review_api_key,
        "enable_review": bool(args.enable_review),
        "review_all": bool(args.review_all),
        "merge_strategy": args.merge_strategy,
        "max_retries": max(int(args.max_retries), 1),
        "timeout": float(args.timeout),
    }


def main() -> int:
    """命令行入口。"""
    setup_logging()
    args = parse_args()

    try:
        runtime = resolve_runtime_config(args)
        input_data = load_results_json(runtime["input_path"])
        raw_items = input_data.get("results", [])

        stem = Path(runtime["input_path"]).stem
        output_dir = Path(runtime["output_dir"])
        ensure_dir(output_dir)
        judged_output_path = output_dir / "{0}_judged.json".format(stem)
        summary_output_path = output_dir / "{0}_judge_summary.json".format(stem)

        judged_items: List[Dict[str, Any]] = []
        total_items = len(raw_items)
        for index, item in enumerate(raw_items, start=1):
            question_id = item.get("question_id", "Q{0:04d}".format(index))
            LOGGER.info("[%s/%s] Judging question_id=%s", index, total_items, question_id)
            try:
                judged_item = judge_single_item(
                    item=item,
                    main_model=runtime["main_model"],
                    main_base_url=runtime["main_base_url"],
                    main_api_key=runtime["main_api_key"],
                    review_model=runtime["review_model"],
                    review_base_url=runtime["review_base_url"],
                    review_api_key=runtime["review_api_key"],
                    enable_review=runtime["enable_review"],
                    review_all=runtime["review_all"],
                    merge_strategy=runtime["merge_strategy"],
                    max_retries=runtime["max_retries"],
                    timeout=runtime["timeout"],
                )
            except Exception as exc:
                LOGGER.exception("Failed to judge question_id=%s: %s", question_id, exc)
                judged_item = dict(item)
                judged_item.update(
                    {
                        "judge_model": runtime["main_model"],
                        "correct": False,
                        "judge_label": "incorrect",
                        "reason": "Unexpected exception during judging: {0}".format(exc),
                        "main_judge_raw": "",
                        "review_judge_raw": "",
                        "final_judge_model": runtime["main_model"],
                        "used_review": False,
                        "review_triggered": False,
                        "main_judge_parsed_ok": False,
                        "review_judge_parsed_ok": False,
                        "main_judge_parse_error": str(exc),
                        "review_judge_parse_error": "",
                    }
                )
            judged_items.append(judged_item)

        summary = compute_accuracy(judged_items)
        summary.update(
            {
                "main_judge_model": runtime["main_model"],
                "review_model": runtime["review_model"],
                "review_enabled": runtime["enable_review"],
            }
        )

        judged_payload = {
            "pkl_name": input_data.get("pkl_name", ""),
            "input_path": str(Path(runtime["input_path"]).resolve()),
            "total": len(judged_items),
            "main_judge_model": runtime["main_model"],
            "review_model": runtime["review_model"],
            "review_enabled": runtime["enable_review"],
            "merge_strategy": runtime["merge_strategy"],
            "judged_results": judged_items,
            "summary": summary,
        }

        save_judged_results(str(judged_output_path), judged_payload)

        summary_payload = {
            "total": summary["total"],
            "num_correct": summary["num_correct"],
            "num_incorrect": summary["num_incorrect"],
            "accuracy": summary["accuracy"],
            "main_judge_model": runtime["main_model"],
            "review_model": runtime["review_model"],
            "review_enabled": runtime["enable_review"],
        }
        save_summary(str(summary_output_path), summary_payload)

        LOGGER.info(
            "Judging finished. total=%s correct=%s incorrect=%s accuracy=%.4f",
            summary["total"],
            summary["num_correct"],
            summary["num_incorrect"],
            summary["accuracy"],
        )
        return 0
    except Exception as exc:
        LOGGER.exception("judge_answers failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
