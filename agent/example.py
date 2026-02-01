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
    print("示例 1: 列出目录内容")
    print("=" * 60)
    print("\n任务: 列出当前目录的所有文件和文件夹")
    print()
    agent.execute_task("列出当前目录的所有文件和文件夹")
    
    print("=" * 60)
    print("示例 2: 读取文件")
    print("=" * 60)
    print("\n任务: 读取 programmer_agent.py 的前15行")
    print()
    agent.execute_task("读取 programmer_agent.py 文件的前15行内容")
    
    print("=" * 60)
    print("示例 3: 创建文件并写入内容")
    print("=" * 60)
    print("\n任务: 在 demo 目录创建 hello.txt 文件并写入欢迎信息")
    print()
    agent.execute_task("在 demo 目录下创建 hello.txt 文件，写入内容'欢迎使用智能编程助手'，然后读取并显示文件内容")
    
    print("=" * 60)
    print("示例 4: 创建目录结构")
    print("=" * 60)
    print("\n任务: 在当前目录创建 project/src 目录结构")
    print()
    agent.execute_task("在当前目录创建 project/src 目录结构")
    
    print("=" * 60)
    print("示例 5: 搜索文件")
    print("=" * 60)
    print("\n任务: 搜索当前目录及其子目录下所有的 .py 文件")
    print()
    agent.execute_task("搜索当前目录及其子目录下所有的 .py 文件并列出路径")
    
    print("=" * 60)
    print("示例 6: 删除文件")
    print("=" * 60)
    print("\n任务: 删除 demo/hello.txt 文件")
    print()
    agent.execute_task("删除 demo/hello.txt 文件")
    
    print("=" * 60)
    print("示例 7: 执行系统命令")
    print("=" * 60)
    print("\n任务: 显示当前工作目录的路径和文件数量")
    print()
    agent.execute_task("显示当前工作目录的路径和其中有多少个文件")
    
    print("=" * 60)
    print("示例 8: 执行 Python 代码")
    print("=" * 60)
    print("\n任务: 计算 1+2+3+...+100 的和并打印结果")
    print()
    agent.execute_task("计算 1+2+3+...+100 的和并打印结果")

def main():
    import os
    import shutil
    
    example_tool_usage()
    
    print("=" * 60)
    print("清理演示文件")
    print("=" * 60)
    for dir_name in ['demo', 'project']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"已删除目录: {dir_name}")

if __name__ == "__main__":
    main()
