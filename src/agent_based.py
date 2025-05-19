"""
agent_based.py

改进版 Materials-AI-Agent
------------------------

支持两种模式：
1) 一次性执行 pipeline：通过 --image_path/--work_root/--user_message 参数启动
2) 交互式 CHAT：省略 --image_path 后进入对话，使用 /run 命令触发

用法示例：
  # 一次性执行
  python agent_based.py \
    --api_key YOUR_KEY \
    --image_path /path/to/img.png \
    --work_root ./results \
    --user_message "元素: Al,Sb，剂量 30k"

  # 交互式
  python agent_based.py --api_key YOUR_KEY
  User> /run /path/to/img.png ./results 元素: Al,Sb

退出 interactive 模式输入 exit/quit。
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI  # pip install openai

# -----------------------------------------------------------------------
# 全局默认（可被 main 覆盖）
# -----------------------------------------------------------------------
_MODEL_NAME = "deepseek-chat"
_PIPELINE_DEFAULTS: Dict[str, str] = {
    "weight_path": "/home/aiprogram/project/yaotian/phase_structure_reconstruction/MOE_model_weights/moe_model.ckpt",
    "label_dir": "/home/aiprogram/project/yaotian/phase_structure_reconstruction/data_generation/label",
    "metadata_csv": "/home/aiprogram/project/yaotian/phase_structure_reconstruction/baseline/property.csv",
}

# -----------------------------------------------------------------------
# 导入主流程函数（请确保 PYTHONPATH 已包含该模块）
# -----------------------------------------------------------------------
try:
    from pipline_framework import run_agent_pipeline
except ImportError:
    print("❌ 无法导入 run_agent_pipeline，请检查 PYTHONPATH 和模块名！")
    sys.exit(1)


# -----------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------
def run_structure_pipeline(
    image_path: str,
    work_root: str,
    user_message: str = "",
) -> str:
    """
    调用完整 STEM → 重建 → shrink → 松弛流程，返回 JSON 字符串。
    """
    img_p = Path(image_path).expanduser().resolve(strict=True)
    root_p = Path(work_root).expanduser()
    root_p.mkdir(parents=True, exist_ok=True)

    try:
        result: Dict[str, Any] = run_agent_pipeline(
            image_path=str(img_p),
            user_message=user_message,
            work_root=str(root_p),
            weight_path=_PIPELINE_DEFAULTS["weight_path"],
            label_dir=_PIPELINE_DEFAULTS["label_dir"],
            metadata_csv=_PIPELINE_DEFAULTS["metadata_csv"],
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        traceback.print_exc()
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


# -----------------------------------------------------------------------
# OpenAI function schema
# -----------------------------------------------------------------------
_FUNCTIONS = [
    {
        "name": "run_structure_pipeline",
        "description": "上传 STEM 大图，自动重建结构并松弛，返回 JSON 结果",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "本地 STEM 图像路径"},
                "work_root": {"type": "string", "description": "结果保存目录"},
                "user_message": {"type": "string", "description": "用户补充说明"},
            },
            "required": ["image_path", "work_root"],
        },
    }
]
_TOOL_MAP = {"run_structure_pipeline": run_structure_pipeline}

# -----------------------------------------------------------------------
# 系统提示
# -----------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "你是 Materials-AI-Agent，能够对 STEM 电镜大图进行结构重建与物性分析。\n"
    "1) 若用户提供大图及工作目录，请模拟调用 run_structure_pipeline，"
    "并以 JSON 格式返回调用参数：\n"
    "   {\"img\": \"<图片路径>\", \"work_root\": \"<工作目录>\", \"user_message\": \"<说明>\"}\n"
    "   然后在下一行输出结构分析结果的专业中文解读（含文件路径、能量、应力等）。\n"
    "2) 若用户只是一般询问，直接回复专业答案。\n"
    "注意：JSON 块必须可被程序用正则提取，不要有额外文字干扰。"
)


# -----------------------------------------------------------------------
# 交互逻辑
# -----------------------------------------------------------------------
def chat(args) -> None:
    # ---------- DeepSeek Chat ----------
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    print("🔹 Materials-AI-Agent 启动，输入 exit/quit 退出。\n")
    print("🟢 提示：在交互中可使用 `/run <图片路径> <工作目录> [说明]` 直接触发结构分析。\n")

    pipeline_executed = False

    while True:
        user_in = input("User> ").strip()
        if user_in.lower() in {"exit", "quit"}:
            break
        if not user_in:
            continue

        # 一次性 /run 命令总是触发流水线
        if user_in.startswith("/run"):
            pipeline_executed = True
            parts = user_in.split(maxsplit=4)
            if len(parts) < 3:
                print("❗ 格式：/run <图片路径> <工作目录> [可选说明]")
                continue
            _, img, wd, *rest = parts
            umsg = rest[0] if rest else ""
            print(f"🔧 立即执行 pipeline: img={img}, work_root={wd}")
            out = run_structure_pipeline(img, wd, umsg)
            print("📁 Pipeline 输出：", out)
            # 回填真实结果并让模型解读
            messages.append({"role": "tool", "name": "run_structure_pipeline", "content": out})
            resp = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=messages,
                temperature=0.2,
            )
            reply = resp.choices[0].message.content
            print(f"\nAssistant> {reply}\n")
            messages.append({"role": "assistant", "content": reply})
            continue

        # 2) 普通对话，但如果已经跑过流水线，就绝不再触发它
        if pipeline_executed:
            # 只走「总结/整理」分支
            messages.append({"role": "user", "content": user_in})
            resp = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=messages,
                temperature=0.2,
            )
            print("\nAssistant>", resp.choices[0].message.content, "\n")
            messages.append(resp.choices[0].message)
            continue

        # 普通对话分支：询问模型是否需要调用流程
        user_prompt = (
            "请判断是否需调用结构分析流程（run_structure_pipeline），"
            "若需要，请首先输出一个纯 JSON 对象，包含键 img, work_root, user_message；\n"
            f"然后在下一行开始请用户耐心等待，并且给出大致所需的时间：\n{user_in}"
        )
        messages.append({"role": "user", "content": user_prompt})

        resp = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=messages,
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        print(f"\nAssistant> {content}\n")
        messages.append(resp.choices[0].message)

        # 尝试从回复中提取 JSON 并触发真实调用
        try:
            json_block = re.search(r"\{[\s\S]*?\}", content).group(0)
            params = json.loads(json_block)
            img = params["img"]
            wd = params["work_root"]
            umsg = params.get("user_message", "")
            print(f"🔧 Detected trigger, 调用 run_structure_pipeline(img={img}, work_root={wd}, user_message={umsg})")
            real_out = run_structure_pipeline(img, wd, umsg)
            pipeline_executed = True
            print("📁 实际 pipeline 返回：", real_out)
            # 再次让模型用真实结果生成解读
            messages2 = messages + [
                {
                    "role": "assistant",                  # 新版 SDK 要么用 "tool"，要么用 "assistant"
                    "name": "run_structure_pipeline",
                    "content": real_out,
                    }
                ]
            resp2 = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=messages2,
                temperature=0.2,
            )
            followup = resp2.choices[0].message.content
            print(f"\nAssistant> {followup}\n")
            messages.append(resp2.choices[0].message)
        except Exception:
            # 无 JSON 或解析失败，则直接当作普通问答
            pass


# -----------------------------------------------------------------------
# Main: 参数 & 模式切换
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Materials-AI-Agent — 一次性或交互式执行 STEM 流水线"
    )
    parser.add_argument("--api_key",     default=None)
    parser.add_argument("--api_base",    default="https://api.deepseek.com")
    parser.add_argument("--model_name",  default=_MODEL_NAME)
    parser.add_argument("--weight_path", default=_PIPELINE_DEFAULTS["weight_path"])
    parser.add_argument("--label_dir",   default=_PIPELINE_DEFAULTS["label_dir"])
    parser.add_argument("--metadata_csv",default=_PIPELINE_DEFAULTS["metadata_csv"])
    parser.add_argument("--image_path",  default=None, help="若指定，则脚本启动后直接执行一次性 pipeline")
    parser.add_argument("--work_root",   default='/home/aiprogram/project/yaotian/phase_structure_reconstruction/baseline', help="一次性 pipeline 的结果保存目录")
    parser.add_argument("--user_message", default=None, help="一次性 pipeline 的用户补充说明")

    args = parser.parse_args()
    # 覆盖配置
    _MODEL_NAME      = args.model_name
    _PIPELINE_DEFAULTS["weight_path"]  = args.weight_path
    _PIPELINE_DEFAULTS["label_dir"]    = args.label_dir
    _PIPELINE_DEFAULTS["metadata_csv"] = args.metadata_csv

    # 一次性模式
    if args.image_path and args.work_root:
        print("🔹 一次性模式：执行 run_structure_pipeline ...")
        out = run_structure_pipeline(args.image_path, args.work_root, args.user_message)
        print("\n🔹 JSON 结果：")
        print(out)
        # 将 out 与用户补充说明一起发送给模型进行解读
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": args.user_message or "请解析以下结果并给出专业解读。"},
            {"role": "assistant", "name": "run_structure_pipeline", "content": out},
        ]
        client = OpenAI(api_key=args.api_key, base_url=args.api_base)

        resp = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=messages,
            temperature=0.2,
        )
        reply = resp.choices[0].message.content
        print(f"\nAssistant> {reply}\n")
        sys.exit(0)

    # 交互式模式
    chat(args)
