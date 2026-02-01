import os
import subprocess
import sys
import json
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass

class ShellCommandTool(Tool):
    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def description(self) -> str:
        return "执行Shell命令，支持运行系统命令和脚本"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的命令"
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间（秒），默认30",
                    "default": 30
                }
            },
            "required": ["command"]
        }

    def execute(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "命令执行超时",
                "stdout": "",
                "stderr": "",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "returncode": -1
            }

class FileReadTool(Tool):
    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "读取文件内容"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "encoding": {
                    "type": "string",
                    "description": "文件编码，默认utf-8",
                    "default": "utf-8"
                }
            },
            "required": ["path"]
        }

    def execute(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            return {
                "success": True,
                "content": content,
                "path": path
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"文件不存在: {path}",
                "content": "",
                "path": path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": "",
                "path": path
            }

class FileWriteTool(Tool):
    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "写入内容到文件"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容"
                },
                "mode": {
                    "type": "string",
                    "description": "写入模式，'w'为覆盖，'a'为追加",
                    "enum": ["w", "a"],
                    "default": "w"
                }
            },
            "required": ["path", "content"]
        }

    def execute(self, path: str, content: str, mode: str = "w") -> Dict[str, Any]:
        try:
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
            return {
                "success": True,
                "message": f"成功写入文件: {path}",
                "path": path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": path
            }

class FileListTool(Tool):
    @property
    def name(self) -> str:
        return "file_list"

    @property
    def description(self) -> str:
        return "列出目录中的文件和文件夹"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录路径，默认当前目录",
                    "default": "."
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "是否显示隐藏文件",
                    "default": False
                }
            },
            "required": ["path"]
        }

    def execute(self, path: str = ".", show_hidden: bool = False) -> Dict[str, Any]:
        try:
            entries = []
            with os.scandir(path) as it:
                for entry in it:
                    if not show_hidden and entry.name.startswith('.'):
                        continue
                    entries.append({
                        "name": entry.name,
                        "is_file": entry.is_file(),
                        "is_dir": entry.is_dir()
                    })
            return {
                "success": True,
                "entries": entries,
                "path": path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entries": [],
                "path": path
            }

class DirectoryCreateTool(Tool):
    @property
    def name(self) -> str:
        return "directory_create"

    @property
    def description(self) -> str:
        return "创建目录"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要创建的目录路径"
                },
                "parents": {
                    "type": "boolean",
                    "description": "是否创建父目录",
                    "default": True
                }
            },
            "required": ["path"]
        }

    def execute(self, path: str, parents: bool = True) -> Dict[str, Any]:
        try:
            if parents:
                os.makedirs(path, exist_ok=True)
            else:
                os.mkdir(path)
            return {
                "success": True,
                "message": f"成功创建目录: {path}",
                "path": path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": path
            }

class FileSearchTool(Tool):
    @property
    def name(self) -> str:
        return "file_search"

    @property
    def description(self) -> str:
        return "搜索文件和文件夹"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "搜索模式，支持通配符"
                },
                "path": {
                    "type": "string",
                    "description": "搜索路径，默认当前目录",
                    "default": "."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "是否递归搜索",
                    "default": True
                }
            },
            "required": ["pattern", "path"]
        }

    def execute(self, pattern: str, path: str = ".", recursive: bool = True) -> Dict[str, Any]:
        try:
            base = Path(path)
            if recursive:
                matches = list(base.rglob(pattern))
            else:
                matches = list(base.glob(pattern))
            
            files = [str(m) for m in matches if m.is_file()]
            dirs = [str(m) for m in matches if m.is_dir()]
            
            return {
                "success": True,
                "files": files,
                "directories": dirs,
                "count": len(files) + len(dirs)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "files": [],
                "directories": [],
                "count": 0
            }

class FileDeleteTool(Tool):
    @property
    def name(self) -> str:
        return "file_delete"

    @property
    def description(self) -> str:
        return "删除文件或目录"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要删除的文件或目录路径"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "删除目录时是否递归",
                    "default": False
                }
            },
            "required": ["path"]
        }

    def execute(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                return {
                    "success": True,
                    "message": f"成功删除文件: {path}",
                    "path": path
                }
            elif os.path.isdir(path):
                if recursive:
                    shutil.rmtree(path)
                else:
                    os.rmdir(path)
                return {
                    "success": True,
                    "message": f"成功删除目录: {path}",
                    "path": path
                }
            else:
                return {
                    "success": False,
                    "error": f"路径不存在: {path}",
                    "path": path
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": path
            }

class PythonExecuteTool(Tool):
    @property
    def name(self) -> str:
        return "python_execute"

    @property
    def description(self) -> str:
        return "执行Python代码片段"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的Python代码"
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间（秒），默认30",
                    "default": 30
                }
            },
            "required": ["code"]
        }

    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "returncode": -1
            }

class ToolManager:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        default_tools = [
            ShellCommandTool(),
            FileReadTool(),
            FileWriteTool(),
            FileListTool(),
            DirectoryCreateTool(),
            FileSearchTool(),
            FileDeleteTool(),
            PythonExecuteTool()
        ]
        
        for tool in default_tools:
            self.register(tool)
    
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> bool:
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self._tools.values()
        ]
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        tool = self.get_tool(tool_name)
        if tool is None:
            return {
                "success": False,
                "error": f"工具不存在: {tool_name}",
                "available_tools": list(self._tools.keys())
            }
        
        try:
            result = tool.execute(**kwargs)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"工具执行失败: {str(e)}",
                "tool": tool_name
            }

def get_tool_descriptions() -> str:
    manager = ToolManager()
    tools = manager.list_tools()
    
    descriptions = ["可用工具列表:\n"]
    for tool in tools:
        desc = f"""
工具名称: {tool['name']}
功能描述: {tool['description']}
参数说明: {json.dumps(tool['parameters'], ensure_ascii=False, indent=6)}
"""
        descriptions.append(desc)
    
    return "\n".join(descriptions)

def get_tool_schema() -> List[Dict[str, Any]]:
    manager = ToolManager()
    return manager.list_tools()
