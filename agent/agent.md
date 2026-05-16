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