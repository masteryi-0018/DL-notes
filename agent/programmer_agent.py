import os
from openai import OpenAI

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
    
    def generate_code(self, task):
        messages = [
            {
                "role": "system",
                "content": "你是一个编程助手。用户会给你一个编程任务，你需要生成Python代码来完成任务。只输出代码，不要解释。"
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
    
    def execute_task(self, task):
        import subprocess
        import sys
        
        code = self.generate_code(task)
        print(f"Generated code:\n{code}\n")
        
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
            print(f"Output:\n{result.stdout}")
            if result.stderr:
                print(f"Errors:\n{result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"Execution failed: {e}")
            return False
        finally:
            if os.path.exists(filename):
                os.remove(filename)

def main():
    agent = SimpleProgrammerAgent()
    
    task = input("请输入编程任务: ")
    agent.execute_task(task)

if __name__ == "__main__":
    main()
