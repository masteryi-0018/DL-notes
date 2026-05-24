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

# ── 工具定义（用最简单的 dict 描述，不搞复杂的类继承）──────────────────

TOOLS = [
    {
        "name": "shell",
        "description": "执行 Shell 命令，返回 stdout/stderr",
        "parameters": {"command": "要执行的命令"},
    },
    {
        "name": "read_file",
        "description": "读取文件内容",
        "parameters": {"path": "文件路径"},
    },
    {
        "name": "write_file",
        "description": "写入文件（覆盖模式）",
        "parameters": {"path": "文件路径", "content": "要写入的内容"},
    },
    {
        "name": "list_dir",
        "description": "列出目录中的文件和子目录",
        "parameters": {"path": "目录路径"},
    },
]


def format_tools():
    """把工具列表转成 prompt 可用的文本"""
    lines = []
    for t in TOOLS:
        params = ", ".join(t["parameters"].keys())
        lines.append(f"  {t['name']}({params}): {t['description']}")
    return "\n".join(lines)


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


# ── 输出解析 ───────────────────────────────────────────────────────

def parse_response(text):
    """解析 LLM 返回的 JSON 工具调用，如果不是工具调用则返回 (tool_call, reasoning) 或 None"""
    text = text.strip()
    # 处理 markdown 代码块包裹的情况
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if lines[0].startswith("```") else text
        if text.endswith("```"):
            text = text[:-3]
    # 尝试找到 JSON 对象（可能前面有推理文字）
    json_start = text.find("{")
    if json_start == -1:
        return None
    reasoning = text[:json_start].strip()
    json_text = text[json_start:]
    try:
        data = json.loads(json_text)
        if "tool" in data and "parameters" in data:
            return (data, reasoning)
    except json.JSONDecodeError:
        pass
    return None


# ── Agent 核心循环 ──────────────────────────────────────────────────

SYSTEM_PROMPT = f"""你是一个编码助手 Agent。你可以使用以下工具来完成用户的编程任务：

{format_tools()}

使用规则：
1. 当你需要做某件事时，必须调用工具。不要假装你能做到——你真的可以！
2. 调用工具时，先用一行简短中文说明你要做什么（为什么选这个工具），然后另起一行，严格按照以下 JSON 格式返回工具调用：

（你的意图说明）
{{"tool": "工具名", "parameters": {{"参数": "值"}}}}

3. 每次只调用一个工具。收到结果后，再决定下一步做什么。
4. 当任务完全完成时，直接输出完成信息（不要再用 JSON 格式）。"""


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
            temperature=0.1,
        )
        reply = resp.choices[0].message.content

        parsed = parse_response(reply)

        # ── 观察 & 行动 ──
        if parsed:
            tool_call, reasoning = parsed
            t_name, t_params = tool_call["tool"], tool_call["parameters"]
            print(f"\n{'─'*50}")
            print(f"[Step {step}] 🤔 Reason: {reasoning}")
            print(f"          🔧 Act:   {t_name}({t_params})")
            result = execute_tool(t_name, t_params)
            print(f"          👁  Observe: {result[:200]}")
            print(f"{'─'*50}")

            # 把这次交互追加到对话历史
            history.append({"role": "assistant", "content": reply})
            history.append({"role": "user", "content": f"工具执行结果:\n{result}"})
        else:
            # ── 没有工具调用，任务完成 ──
            print(f"\n{'='*50}")
            print(f"[Step {step}] ✅ Agent 认为任务完成:\n")
            print(reply)
            print(f"{'='*50}")
            return

    print("\n⚠️ 达到最大步数限制，Agent 停止。")


# ── 入口 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "GLM_API_KEY" not in os.environ:
        print("请先设置环境变量: export GLM_API_KEY='your-key'")
        sys.exit(1)

    print("编码 Agent Demo —— 展示 Agent 核心循环")
    print("可用的工具: " + ", ".join(t["name"] for t in TOOLS))
    task = input("\n请输入任务: ").strip()
    if task:
        agent_loop(task)
