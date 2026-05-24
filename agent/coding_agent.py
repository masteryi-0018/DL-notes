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

# ── 工具注册中心（MCP 思想的最简表达：工具定义和执行与 Agent 解耦）──

class ToolRegistry:
    """工具注册中心：注册工具定义 + 执行工具调用"""

    def __init__(self):
        self._tools = {}

    def register(self, name, description, parameters, handler):
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
            "handler": handler,
        }

    def list_tools(self):
        """返回 OpenAI Function Calling 格式的工具列表"""
        return [{"type": t["type"], "function": t["function"]} for t in self._tools.values()]

    def call_tool(self, name, arguments):
        """执行工具并返回结果文本"""
        handler = self._tools[name]["handler"]
        try:
            return handler(**arguments)
        except Exception as e:
            return f"执行失败: {e}"


registry = ToolRegistry()

registry.register(
    "shell", "执行 Shell 命令，返回 stdout/stderr",
    {"type": "object", "properties": {"command": {"type": "string", "description": "要执行的命令"}}, "required": ["command"]},
    lambda command: subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30).stdout or "(无输出)",
)

registry.register(
    "read_file", "读取文件内容",
    {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}}, "required": ["path"]},
    lambda path: open(path).read(),
)

registry.register(
    "write_file", "写入文件（覆盖模式）",
    {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "要写入的内容"}}, "required": ["path", "content"]},
    lambda path, content: (open(path, "w").write(content), "写入成功")[1],
)

registry.register(
    "list_dir", "列出目录中的文件和子目录",
    {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径"}}, "required": ["path"]},
    lambda path: "\n".join(os.listdir(path)) if os.listdir(path) else "(空目录)",
)


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

    tools = registry.list_tools()

    for step in range(1, max_steps + 1):
        # ── 思考：LLM 决定下一步做什么 ──
        resp = client.chat.completions.create(
            model="glm-4.7-flash",
            messages=history,
            tools=tools,
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
            result = registry.call_tool(t_name, t_args)
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
    print("可用的工具: " + ", ".join(t["function"]["name"] for t in registry.list_tools()))
    task = input("\n请输入任务: ").strip()
    if task:
        agent_loop(task)
