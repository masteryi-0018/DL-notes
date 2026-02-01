from programmer_agent import SimpleProgrammerAgent

def example_tool_usage():
    agent = SimpleProgrammerAgent()
    
    print("=" * 60)
    print("智能编程助手 - 工具调用示例")
    print("=" * 60)
    
    print("\n可用工具:")
    for tool in agent.tool_manager.list_tools():
        print(f"  • {tool['name']}: {tool['description']}")
    print()
    
    print("=" * 60)
    print("示例 1: 查看项目结构")
    print("=" * 60)
    print("\n任务: 列出当前目录的文件和文件夹")
    print()
    agent.execute_task_with_tools("列出当前目录的文件和文件夹")
    
    print("\n" + "=" * 60)
    print("示例 2: 读取文件内容")
    print("=" * 60)
    print("\n任务: 读取 programmer_agent.py 的前20行代码")
    print()
    agent.execute_task_with_tools("读取 programmer_agent.py 文件的前20行内容并显示")
    
    print("=" * 60)
    print("示例 3: 搜索文件")
    print("=" * 60)
    print("\n任务: 搜索并列出所有 Python 文件")
    print()
    agent.execute_task_with_tools("搜索当前目录下所有的 Python 文件并列出它们的路径")
    
    print("\n" + "=" * 60)
    print("示例 4: 创建目录")
    print("=" * 60)
    print("\n任务: 在 /tmp 目录下创建新的项目目录")
    print()
    agent.execute_task_with_tools("在 /tmp 目录下创建一个名为 my_project 的新目录")
    
    print("=" * 60)
    print("示例 5: 写入文件")
    print("=" * 60)
    print("\n任务: 创建并写入一个包含欢迎信息的消息文件")
    print()
    agent.execute_task_with_tools("在 /tmp/my_project 目录下创建 message.txt 文件，写入内容'欢迎使用智能编程助手'，然后读取并显示文件内容")
    
    print("\n" + "=" * 60)
    print("示例 6: 执行系统命令")
    print("=" * 60)
    print("\n任务: 显示当前工作目录和目录内容")
    print()
    agent.execute_task_with_tools("使用 shell_command 工具执行命令，显示当前工作目录和目录内容")
    
    print("=" * 60)
    print("示例 7: 执行 Python 代码")
    print("=" * 60)
    print("\n任务: 计算并显示斐波那契数列的第15项")
    print()
    agent.execute_task_with_tools("使用 python_execute 工具计算斐波那契数列的第15项并打印结果")
    
    print("\n" + "=" * 60)
    print("示例 8: 删除文件")
    print("=" * 60)
    print("\n任务: 删除刚才创建的文件和目录")
    print()
    agent.execute_task_with_tools("删除 /tmp/my_project/message.txt 文件和 /tmp/my_project 目录")
    
    print("\n" + "=" * 60)
    print("示例 9: 综合任务")
    print("=" * 60)
    print("\n任务: 创建一个完整的项目结构并运行代码")
    print()
    agent.execute_task_with_tools("在 /tmp/demo_project 目录下创建以下结构：创建 src 目录，在其中创建 app.py 文件，写入计算1到100之和的代码并执行")

def main():
    example_tool_usage()

if __name__ == "__main__":
    main()
