#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystal images ➜ multi-LLM batch inference  (rev-2025-05-01)
-----------------------------------------------------------
关键改动：
1. 统一 logging；环境变量缺失立即报错并退出
2. 自动推断图片 MIME，避免硬编码 image/jpeg
3. OpenAI SDK 旧/新版本兼容调用
4. DashScope / 智谱接口错误时安全解析，返回 None
5. parse_model_output 先正则提取首段 JSON，空值直接短路
6. 主流程捕获异常后继续下一张图，同时把 error_type 记录下来
"""
import os
import re
import json
import base64
import mimetypes
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

# ─────────────────────────── 第三方 SDK ──────────────────────────── #
try:
    import openai
except ImportError:
    openai = None

try:
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None  # noqa: N816

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None
# ------------------------------------------------------------------ #

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


# ╔═══════════════════════════ 工具函数 ════════════════════════════╗
def get_image_files(directory: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".gif")) -> List[Path]:
    return [Path(directory) / f for f in os.listdir(directory) if f.lower().endswith(exts)]


def require_api_key(name: str, default: Optional[str] = None) -> str:
    key = os.getenv(name, default)
    if not key:
        logging.critical("Environment variable %s not set!", name)
        raise SystemExit(1)
    return key


def guess_mime_type(fp: Path) -> str:
    mt, _ = mimetypes.guess_type(fp.name)
    return mt or "application/octet-stream"


def encode_image_to_base64(fp: Path) -> str:
    return base64.b64encode(fp.read_bytes()).decode("utf-8")


# ╔══════════════════════════─ API 调用封装 ─═══════════════════════╗
def call_dashscope(api_key: str, model: str, image_fp: Path, prompt: str) -> Optional[str]:
    if MultiModalConversation is None:  # local env 没装 dashscope
        logging.error("dashscope SDK not installed")
        return None
    image_uri = f"file://{image_fp.resolve()}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"image": image_uri},
                {"text": prompt.rstrip()},
            ],
        },
    ]
    try:
        rsp: Dict[str, Any] = MultiModalConversation.call(
            api_key=api_key, model=model, messages=messages
        )
        content = (
            rsp.get("output", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", [{}])
        )
        # content 可能是 list[dict] 或 str
        if isinstance(content, list):
            return content[0].get("text")
        return content
    except Exception as exc:
        logging.error("DashScope error: %s", exc)
        return None


def call_openai(api_key: str, model: str, image_fp: Path, prompt: str) -> Optional[str]:
    if openai is None:
        logging.error("openai SDK not installed")
        return None
    openai.api_key = api_key
    mime = guess_mime_type(image_fp)
    image_b64 = encode_image_to_base64(image_fp)
    image_url = f"data:{mime};base64,{image_b64}"

    try:
        # 兼容 openai==1.x 与 0.x
        if hasattr(openai, "chat"):  # >=1.0.0
            chat = openai.chat.completions.create
            rsp = chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.rstrip()},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            )
            return rsp.choices[0].message.content
        # fallback 到 0.x
        rsp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.rstrip()},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )
        return rsp.choices[0].message["content"]
    except Exception as exc:
        logging.error("OpenAI error: %s", exc)
        return None


def call_zhipuai(api_key: str, model: str, image_fp: Path, prompt: str) -> Optional[str]:
    if ZhipuAI is None:
        logging.error("ZhipuAI SDK not installed")
        return None
    mime = guess_mime_type(image_fp)
    image_b64 = encode_image_to_base64(image_fp)
    url = f"data:{mime};base64,{image_b64}"

    try:
        client = ZhipuAI(api_key=api_key)
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": prompt.rstrip()},
                    ],
                }
            ],
        )
        msg = rsp.choices[0].message
        # 新版 SDK ⇒ msg.content，旧版 ⇒ msg.text
        return getattr(msg, "content", None) or getattr(msg, "text", None)
    except Exception as exc:
        logging.error("ZhipuAI error: %s", exc)
        return None


# ╔═══════════════════════ 解析 JSON 输出 ═════════════════════════╗
_JSON_RE = re.compile(r"\{.*}", re.S)


def parse_model_output(text: Optional[str]) -> Tuple[Optional[str], ...]:
    """始终返回 (crystal_type, energy_above_hull, formation_energy, band_gap, error_type)"""
    if not text:
        return None, None, None, None, "empty_output"

    m = _JSON_RE.search(text)
    if not m:
        return None, None, None, None, "json_not_found"

    try:
        data = json.loads(m.group(0))
        return (
            data.get("crystal_type"),
            data.get("energy_above_hull"),
            data.get("formation_energy"),
            data.get("band_gap"),
            None,  # error_type
        )
    except json.JSONDecodeError:
        return None, None, None, None, "json_decode_error"


# ╔══════════════════════ 主批处理逻辑 ════════════════════════════╗
def test_models_on_images(
    image_dir: str,
    prompt: str,
    models_cfg: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    imgs = get_image_files(image_dir)
    if not imgs:
        logging.warning("No images found under %s", image_dir)
        return []

    results: List[Dict[str, Any]] = []
    for cfg in models_cfg:
        model, platform, api_key = cfg["name"], cfg["platform"].lower(), cfg["api_key"]
        logging.info("▶ Testing model=%s  platform=%s", model, platform)

        # 若未指定 api_key，立即终止
        api_key = require_api_key(api_key) if not api_key.startswith("${") else require_api_key(api_key[2:-1])

        for img in imgs:
            logging.info("   processing %-40s", img.name)
            if platform == "dashscope":
                raw = call_dashscope(api_key, model, img, prompt)
            elif platform == "openai":
                raw = call_openai(api_key, model, img, prompt)
            elif platform == "zhipuai":
                raw = call_zhipuai(api_key, model, img, prompt)
            else:
                logging.error("Unknown platform %s", platform)
                continue

            ct, e_hull, f_energy, gap, err = parse_model_output(raw)
            results.append(
                dict(
                    model=model,
                    platform=platform,
                    image_file=str(img),
                    crystal_type=ct,
                    energy_above_hull=e_hull,
                    formation_energy=f_energy,
                    band_gap=gap,
                    error_type=err,
                    raw_output=raw,
                )
            )
    return results


def save_results_csv(rows: List[Dict[str, Any]], csv_fp: str = "crystal_analysis_results.csv") -> None:
    if not rows:
        logging.warning("No rows to save.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(csv_fp, index=False, encoding="utf-8")
    logging.info("Results saved to %s (%d rows)", csv_fp, len(df))


# ╔══════════════════════════─ main ─═════════════════════════════╗
if __name__ == "__main__":
    IMAGE_DIR = "pic"  # ⇽ 按需修改
    PROMPT = (
        "你将收到一张晶体结构图像，请分析该结构的晶体类型，并根据图像信息估算下列参数：\n"
        "- crystal_type\n- energy_above_hull (eV/atom)\n- formation_energy (eV/atom)\n- band_gap (eV)\n"
        "请仅以最小化后的 JSON 格式返回，不要包含解释性文字。"
    )

    MODELS = [
        {
            "name": "qwen-vl-max-latest",
            "platform": "dashscope",
            "api_key": "DASHSCOPE_API_KEY",  # 环境变量名
        },
        {
            "name": "gpt-4o",
            "platform": "openai",
            "api_key": "OPENAI_API_KEY",
        },
        # 如需智谱：
        # {
        #     "name": "glm-4v",
        #     "platform": "zhipuai",
        #     "api_key": "ZHIPU_API_KEY",
        # },
    ]

    all_rows = test_models_on_images(IMAGE_DIR, PROMPT, MODELS)
    save_results_csv(all_rows)
