# 科学上网

## 大致流程

1. 购买境外服务器
2. 使用连接工具搭建服务
3. 开始使用

## 详细介绍

### 境外服务器（VPS）

1. <https://www.vultr.com/zh/>
2. <https://bwh89.net/cart.php?gid=1>，最便宜的大概 50$/year

### 连接工具（tunnel proxy）

1. <https://github.com/shadowsocks>
    - 电脑端
    - 手机端，Google play可以下载

2. <https://github.com/v2fly>，本来的v2ray；
    - V2Ray / V2Fly 是最新的黑科技加持的技术，相比 Shadowsocks 来说更加安全，高效，隐蔽
    - <https://github.com/2dust/v2rayN>，A GUI client for Windows【重要】，同作者还有其他的工具，原理一样

3. OPENVPN
    - 客户端不好用

4. <https://github.com/yichengchen/clashX>，一个系列，GitHub直接搜就好
    - 最近Windows版的GUI被封了...

## 教程

1. <https://github.com/233boy/v2ray>，一键安装脚本
2. 同一个作者，Xray，<https://github.com/233boy/Xray>

## 机场

我们一般将提供 v2ray / v2fly 代理或 shadowsocks 服务的商家，俗称机场，或者飞机场

1. 搬瓦工 JMS 机场，Just My Socks，<https://justmysocks5.net/members/cart.php?gid=1&language=chinese>
2. clashX使用的，<https://2.akkcloud1.com/auth/login>

## 虚拟信用卡

- <https://bewildcard.com/>，就是感觉有点贵

## 关于ChatGPT

有一些是VPS的ip是被打入了黑名单的，上去会提示ip是不被允许访问的，有一些已知的未被限制的

- 搬瓦工 JMS 机场：解锁 ChatGPT 访问

JMS 速度：香港 > 日本 > 洛杉矶 > 伦敦
VPS 速度：香港线路 > 日本线路 > CN2 GIA 线路 > CN2 线路 > 普通线路

## 关于GitHub

- [镜像、host](https://github.com/Alvin9999/new-pac/wiki/%E4%B8%8D%E7%BF%BB%E5%A2%99%E4%B8%8Agithub)

## 关于conda和pip

### conda

```
vim ~/.condarc

channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - defaults
show_channel_urls: true
```

这样依然会导致去默认channel搜索，不能完全解决问题，可以采用以下的方法：

```
vim ~/.condarc

channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
ssl_verify: true
```

### pip

```
echo "[global]" > ~/pip.conf
echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/pip.conf
mkdir .pip
mv ~/pip.conf ~/.pip/pip.conf
```