## claude+deepseek配置

1. 下载claude code

有vpn：
```sh
curl -fsSL https://claude.ai/install.sh | bash
```

无vpn：
```sh
sudo apt install npm

# 通过nvm安装
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# 官方直接的安装方式不行，得用这个deprecated的方式
npm install -g @anthropic-ai/claude-code
```

无vpn网也卡：
```sh
sudo apt install npm

# nvm镜像
curl -o- https://gitee.com/mirrors/nvm/raw/v0.40.3/install.sh | bash

# 设置Node.js下载镜像
export NVM_NODEJS_ORG_MIRROR=http://mirrors.cloud.tencent.com/nodejs-release/

nvm install 18
npm install -g @anthropic-ai/claude-code
```

2. 解除地区限制

```json
"hasCompletedOnboarding": true
```

3. 添加deepseek

```json
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的key",
    "ANTHROPIC_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-v4-flash",
    "CLAUDE_CODE_EFFORT_LEVEL": "max"
  },
```

## LLM、Agent、JSON解析、Function Calling、MCP 的关系

```
                        ┌───────────────────────────────────────┐
                        │               Agent                    │
                        │                                       │
  用户任务 ──────────────→                                   │
                        │   ┌──────────┐                       │
                        │   │   LLM    │                       │
                        │   │ (大脑)   │                       │
                        │   └────┬─────┘                       │
                        │        │ 驱动                         │
                        │   ┌────▼──────────────────────┐      │
                        │   │  Agent 循环                │      │
                        │   │  Reason → Act → Observe   │      │
                        │   └──┬──────────────┬─────────┘      │
                        │      │              │                 │
                        └──────┼──────────────┼─────────────────┘
                               │              │
                    决定调用什么工具      返回结果给循环
                               │              │
                    ┌──────────▼──┐      ┌───┴──────────┐
                    │  工具调用方式 │      │   工具调用方式  │   ← "LLM 怎么表达调用意图"
                    │             │      │              │
                    │  JSON 解析   │      │ Function     │
                    │             │      │ Calling      │
                    │ prompt里写   │      │              │
                    │ 工具描述     │      │ tools参数传   │
                    │ LLM输出JSON  │      │ schema       │
                    │ Agent手工解析 │      │ LLM返回      │
                    │             │      │ tool_calls   │
                    └──────┬──────┘      └──────┬───────┘
                           │                    │
                           │    最终都是：       │
                           │    "调用工具 X"     │
                           └─────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │        MCP           │   ← "工具怎么注册、怎么通信"
                          │                      │
                          │  list_tools()        │   工具发现
                          │  call_tool(name,args)│   工具调用
                          │                      │
                          │  有 MCP：工具在哪实现 │
                          │  Agent 不关心，       │
                          │  统一接口调用         │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼──────────┐
                          │      工具实现         │
                          │                      │
                          │  shell  read_file   │
                          │  write_file  list_dir│
                          └──────────────────────┘
```

三层抽象，各解决各的问题：

  第 1 层：Agent 循环          ← 什么时候调用、结果怎么处理
  第 2 层：JSON 解析 / FC      ← LLM 怎么表达「我要调用哪个工具」
  第 3 层：MCP                 ← 工具怎么注册、怎么通信、Agent 怎么发现工具

  JSON 解析：优点是不依赖 API 能力，任何模型都能用；缺点是解析脆弱，格式不稳定
  Function Calling：优点是格式可靠，随 API 标准化；缺点是需要模型支持
  MCP：优点是工具和 Agent 解耦，跨语言跨进程复用工具；缺点是多一层抽象，增加复杂度

  没有 MCP 时：Agent 直接调用工具（工具写死在代码里）
  有 MCP 时：Agent 通过 MCP 发现和调用工具（工具可以独立开发部署）

  这三层是正交的——JSON解析/FC 和 MCP 可以任意组合：
    JSON解析 + 无MCP（最早的手工方式）
    JSON解析 + MCP   （模型不支持FC时的方案）
    FC + 无MCP       （最简方案，coding_agent.py 的 MCP 提交前）
    FC + MCP         （标准化方案，coding_agent.py 当前版本）