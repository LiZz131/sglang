#!/usr/bin/env python3
"""飞书群机器人 Webhook 发一条文本通知。适合在 bash 脚本末尾调用。

用法:
  export FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/...'
  python3 send_feishu_message.py                    # 默认「任务已结束」
  python3 send_feishu_message.py "nsys 采集完成"     # 自定义文案

也可只设环境变量 FEISHU_MESSAGE 覆盖默认正文（无命令行参数时生效）。
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

import requests


def send_feishu_text(webhook_url: str, text: str, title: str = "任务通知") -> None:
    payload = {
        "msg_type": "text",
        "content": {"text": f"【{title}】\n{text}"},
    }
    res = requests.post(webhook_url, json=payload, timeout=15)
    res.raise_for_status()
    try:
        body = res.json() if res.content else {}
    except Exception:
        body = {}
    # 飞书机器人成功时 code 常为 0；无 code 字段则视为成功
    if isinstance(body, dict) and "code" in body and body.get("code") != 0:
        raise RuntimeError(f"飞书 API 返回异常: {body}")


def main() -> int:
    parser = argparse.ArgumentParser(description="通过飞书 Webhook 发送文本消息")
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="消息正文；省略则用环境变量 FEISHU_MESSAGE 或默认「任务已结束」",
    )
    parser.add_argument(
        "--title",
        default=os.environ.get("FEISHU_TITLE", "GPU/脚本任务"),
        help="标题前缀（也可用环境变量 FEISHU_TITLE）",
    )
    args = parser.parse_args()

    url = os.environ.get("FEISHU_WEBHOOK_URL", "").strip()
    if not url:
        print("错误: 请设置环境变量 FEISHU_WEBHOOK_URL", file=sys.stderr)
        return 2

    if args.message is not None:
        body = args.message
    else:
        body = os.environ.get("FEISHU_MESSAGE", "").strip() or "任务已结束"

    host = socket.gethostname()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full = f"{body}\n\n主机: {host}\n时间: {now}"

    try:
        send_feishu_text(url, full, title=args.title)
    except Exception as e:
        print(f"发送失败: {e}", file=sys.stderr)
        return 1

    print("已发送飞书通知。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
