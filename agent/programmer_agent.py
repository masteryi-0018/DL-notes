import os
import json
from openai import OpenAI
from tools import ToolManager, get_tool_descriptions

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), ".env")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key] = value
    return config

class SimpleProgrammerAgent:
    def __init__(self, api_key=None, model=None):
        config = load_config()
        
        self.api_key = api_key or os.environ.get("GLM_API_KEY") or config.get("GLM_API_KEY")
        self.model = model or config.get("MODEL_NAME", "glm-4")
        
        if not self.api_key:
            raise ValueError("API key not found. Please set GLM_API_KEY in .env file or environment variables.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        
        self.tool_manager = ToolManager()
    
    def _get_system_prompt(self):
        tool_descriptions = get_tool_descriptions()
        return f"""你是一个智能编程助手，拥有使用各种工具的能力。

{tool_descriptions}

使用规则：
1. 当需要执行系统命令时，使用 shell_command 工具
2. 当需要读取文件时，使用 file_read 工具
3. 当需要写入文件时，使用 file_write 工具
4. 当需要列出目录内容时，使用 file_list 工具
5. 当需要创建目录时，使用 directory_create 工具
6. 当需要搜索文件时，使用 file_search 工具
7. 当需要删除文件或目录时，使用 file_delete 工具
8. 当需要执行Python代码时，使用 python_execute 工具

如果需要使用工具，请按照以下JSON格式返回工具调用请求：
{{
    "tool": "工具名称",
    "parameters": {{
        "参数名": "参数值"
    }}
}}

如果不需要使用工具，请直接返回代码或回答。
只输出代码或JSON，不要有多余的解释。"""
    
    def generate_code(self, task, with_tools=False):
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt() if with_tools else "你是一个编程助手。用户会给你一个编程任务，你需要生成Python代码来完成任务。只输出代码，不要解释。"
            },
            {
                "role": "user",
                "content": task
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def clean_code(self, code):
        lines = code.strip().split('\n')
        cleaned_lines = []
        
        in_code_block = False
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                cleaned_lines.append(line)
            elif not line.strip().startswith('```'):
                cleaned_lines.append(line)
        
        cleaned_code = '\n'.join(cleaned_lines).strip()
        
        if not cleaned_code:
            cleaned_code = code.strip()
        
        return cleaned_code
    
    def _parse_tool_call(self, response_text):
        try:
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                for i, line in enumerate(lines):
                    if '```' in line and i > 0:
                        json_text = '\n'.join(lines[i+1:])
                        if json_text.endswith('```'):
                            json_text = json_text[:-3]
                        json_text = json_text.strip()
                        break
                else:
                    json_text = response_text
            else:
                json_text = response_text
            
            if json_text.startswith('{') and json_text.endswith('}'):
                tool_call = json.loads(json_text)
                if "tool" in tool_call and "parameters" in tool_call:
                    return tool_call
        except json.JSONDecodeError:
            pass
        return None
    
    def _execute_tool_call(self, tool_call):
        tool_name = tool_call.get("tool")
        parameters = tool_call.get("parameters", {})
        
        tool = self.tool_manager.get_tool(tool_name)
        if tool:
            purpose = tool.description
        else:
            purpose = "未知工具"
        
        print(f"\n{'='*60}")
        print(f"🤖 AI 决定调用工具: {tool_name}")
        print(f"📋 工具功能: {purpose}")
        print(f"📝 参数: {json.dumps(parameters, ensure_ascii=False, indent=2)}")
        print(f"{'='*60}")
        
        result = self.tool_manager.execute(tool_name, **parameters)
        
        print(f"\n✅ 工具执行结果:")
        if result.get("success"):
            if "stdout" in result and result["stdout"]:
                print(result["stdout"])
            if "content" in result and result["content"]:
                preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                print(f"内容预览: {preview}")
            if "files" in result:
                print(f"找到文件: {result['files']}")
            if "entries" in result:
                print(f"目录条目: {len(result['entries'])} 个")
            print(f"状态: 成功")
        else:
            print(f"❌ 错误: {result.get('error', '未知错误')}")
        print(f"{'='*60}\n")
        
        return result
    
    def execute_task_with_tools(self, task, max_iterations=5):
        print(f"{'='*60}")
        print(f"🎯 开始执行任务")
        print(f"{'='*60}")
        print(f"任务描述: {task}")
        print(f"{'='*60}\n")
        
        iteration = 0
        conversation_history = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": task
            }
        ]
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n📍 步骤 {iteration}/{max_iterations}: AI 正在思考...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation_history,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            
            tool_call = self._parse_tool_call(assistant_response)
            
            if tool_call:
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                tool_result = self._execute_tool_call(tool_call)
                
                conversation_history.append({
                    "role": "user",
                    "content": f"工具执行结果: {json.dumps(tool_result, ensure_ascii=False, indent=2)}\n\n请根据结果继续执行任务。如果任务已完成，请返回最终代码或结果。"
                })
            else:
                if "```python" in assistant_response or assistant_response.startswith("def ") or assistant_response.startswith("class ") or assistant_response.startswith("import "):
                    print(f"\n✨ AI 完成任务，返回代码:")
                    print(f"{'='*60}")
                    print(assistant_response)
                    print(f"{'='*60}\n")
                    return True
                else:
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    conversation_history.append({
                        "role": "user",
                        "content": "请提供具体的代码实现或继续使用工具完成任务。"
                    })
        
        print("⚠️ 达到最大迭代次数，任务未完成")
        return False

def main():
    agent = SimpleProgrammerAgent()
    
    print("=" * 60)
    print("智能编程助手 - 支持工具调用")
    print("=" * 60)
    
    print("\n可用工具:")
    for tool in agent.tool_manager.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\n" + "=" * 60)
    task = input("请输入编程任务: ")
    print("=" * 60)
    
    print("\n选择执行模式:")
    print("1. 简单模式（仅生成和执行代码）")
    print("2. 工具模式（可使用各种工具）")
    mode = input("请选择 (1/2): ").strip()
    
    if mode == "2":
        agent.execute_task_with_tools(task)
    else:
        code = agent.generate_code(task)
        print(f"\n生成的代码:\n{code}\n")
        
        import subprocess
        import sys
        
        filename = "temp_agent_code.py"
        with open(filename, "w") as f:
            f.write(code)
        
        try:
            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=30
            )
            print(f"输出:\n{result.stdout}")
            if result.stderr:
                print(f"错误:\n{result.stderr}")
        except Exception as e:
            print(f"执行失败: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    main()
