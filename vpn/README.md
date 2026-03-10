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

## 代理学习笔记

1. 目前使用的科学上网工具，是通过代理实现的。点击连接后，电脑的代理服务器会被设置，设置的代理服务器是本机，端口是10808，这个端口就是由科学上网的工具创建的，工具会监听这个端口并处理请求。也就是说，访问国外网站时，电脑会将请求发送到本机的10808端口，然后这个代理服务器会将请求转发到境外服务器，再由境外服务器访问国外网站并将结果返回给电脑。
2. 电脑在自己的局域网内的ip是192.168.1.5，路由器的ip是192.168.1.1，但当电脑创建热点时，电脑的ip就变成了手机的路由器ip，比如我这里是192.168.137.1，手机连接这个热点，手机的路由器ip就变成了这个。
3. （可选）但是发现手机连接这个热点后，微信和币站可以使用，但是豆包和网页都不能使用，是因为“私有无线局域网地址”这个选项决定的，包括关闭、固定和轮替，默认是轮替，但是会导致电脑作为路由器混乱无法处理，但关闭会导致手机的唯一MAC地址暴露，不安全，所以可以修改为固定，设置一个虚假的固定的MAC地址，这样就可以正常使用了。
4. DNS默认就是电脑的ip。手机要想科学上网，可以设置代理为电脑ip的10808端口，但是目前设置代理就会导致无法使用浏览器。原因是电脑不会讲手机过来的流量转发到本机的特定端口，需要写一个脚本进行端口转发，之后手机也可以科学上网了。
5. 操作系统设置了代理，实际上是告诉上层，“我这有一份代理建议，你要不要用？”，上层可以选择使用这个代理（使用系统网络api），也可以选择不使用这个代理（自己设置网络）。所以浏览器可以访问YouTube，但是app不能连接，因为app可能没走代理。