from programmer_agent import SimpleProgrammerAgent

def example_tasks():
    agent = SimpleProgrammerAgent()
    
    tasks = [
        "编写一个函数计算斐波那契数列的第n项",
        "创建一个函数反转字符串",
        "编写一个简单的计算器可以加减乘除"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*50}")
        print(f"任务 {i}: {task}")
        print('='*50)
        agent.execute_task(task)

if __name__ == "__main__":
    example_tasks()
