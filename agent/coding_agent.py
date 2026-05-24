"""
一个极简的编码 Agent 演示，展示 Agent 的四个核心概念：
  1. LLM 驱动 —— 大模型作为大脑
  2. 工具调用 —— Agent 能操作外部世界
  3. Agent 循环 —— 思考→行动→观察→再思考，直到任务完成
  4. System Prompt —— 定义 Agent 的行为边界

使用方式：
  export GLM_API_KEY="your-key"
  python coding_agent.py
"""

import os, json, subprocess, sys

from openai import OpenAI

# ── 工具定义（OpenAI Function Calling 格式）────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "执行 Shell 命令，返回 stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件（覆盖模式）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录中的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径"}
                },
                "required": ["path"],
            },
        },
    },
]


# ── 工具执行 ───────────────────────────────────────────────────────

def execute_tool(name, params):
    """根据工具名分发执行，返回结果字符串"""
    try:
        if name == "shell":
            r = subprocess.run(params["command"], shell=True,
                               capture_output=True, text=True, timeout=30)
            return r.stdout or r.stderr or "(无输出)"
        elif name == "read_file":
            with open(params["path"]) as f:
                return f.read()
        elif name == "write_file":
            with open(params["path"], "w") as f:
                f.write(params["content"])
            return "写入成功"
        elif name == "list_dir":
            entries = os.listdir(params["path"])
            return "\n".join(entries) if entries else "(空目录)"
        else:
            return f"未知工具: {name}"
    except Exception as e:
        return f"执行失败: {e}"


# ── Agent 核心循环 ──────────────────────────────────────────────────

SYSTEM_PROMPT = f"""你是一个编码助手 Agent。你可以使用工具来完成用户的编程任务。

使用规则：
1. 当你需要做某件事时，必须调用工具。不要假装你能做到——你真的可以！
2. 每次只调用一个工具。收到结果后，再决定下一步做什么。
3. 当任务完全完成时，直接输出完成信息。"""


def agent_loop(task, max_steps=10):
    """Agent 的主循环：每一步都是 思考→行动→观察 的循环"""
    client = OpenAI(
        api_key=os.environ["GLM_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        # ── 思考：LLM 决定下一步做什么 ──
        resp = client.chat.completions.create(
            model="glm-4.7-flash",
            messages=history,
            tools=TOOLS,
            temperature=0.1,
        )
        msg = resp.choices[0].message

        # ── 观察 & 行动 ──
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            t_name = tc.function.name
            t_args = json.loads(tc.function.arguments)

            print(f"\n{'─'*50}")
            print(f"[Step {step}] 🤔 Reason: {msg.content or '(模型决定调用工具)'}")
            print(f"          🔧 Act:   {t_name}({json.dumps(t_args, ensure_ascii=False)})")
            result = execute_tool(t_name, t_args)
            print(f"          👁  Observe: {result[:200]}")
            print(f"{'─'*50}")

            # 把这次交互追加到对话历史
            history.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": t_name, "arguments": tc.function.arguments},
                    }
                ],
            })
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        else:
            # ── 没有工具调用，任务完成 ──
            print(f"\n{'='*50}")
            print(f"[Step {step}] ✅ Agent 认为任务完成:\n")
            print(msg.content)
            print(f"{'='*50}")
            return

    print("\n⚠️ 达到最大步数限制，Agent 停止。")


# ── 入口 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "GLM_API_KEY" not in os.environ:
        print("请先设置环境变量: export GLM_API_KEY='your-key'")
        sys.exit(1)

    print("编码 Agent Demo —— 展示 Agent 核心循环")
    print("可用的工具: " + ", ".join(t["function"]["name"] for t in TOOLS))
    task = input("\n请输入任务: ").strip()
    if task:
        agent_loop(task)
