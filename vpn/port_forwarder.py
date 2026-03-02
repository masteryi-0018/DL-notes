#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import threading
import sys
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortForwarder:
    """
    端口转发器 - 将流量从一个地址:端口转发到另一个地址:端口
    """

    def __init__(self, listen_host, listen_port, target_host, target_port):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.target_host = target_host
        self.target_port = target_port
        self.server_socket = None
        self.running = False

    def start(self):
        """启动端口转发服务器"""
        try:
            # 创建监听socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 绑定地址和端口
            self.server_socket.bind((self.listen_host, self.listen_port))
            self.server_socket.listen(100)  # 最大连接数
            self.running = True

            logger.info(f"端口转发服务已启动")
            logger.info(f"监听地址: {self.listen_host}:{self.listen_port}")
            logger.info(f"转发目标: {self.target_host}:{self.target_port}")
            logger.info("等待连接...")

            while self.running:
                try:
                    # 接受客户端连接
                    client_socket, client_addr = self.server_socket.accept()
                    logger.info(f"接受连接来自: {client_addr[0]}:{client_addr[1]}")

                    # 为每个连接创建新线程处理
                    thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_addr)
                    )
                    thread.daemon = True
                    thread.start()

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"接受连接时出错: {e}")

        except Exception as e:
            logger.error(f"启动端口转发服务失败: {e}")
        finally:
            self.stop()

    def handle_client(self, client_socket, client_addr):
        """处理客户端连接"""
        target_socket = None
        try:
            # 连接到目标服务器
            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_socket.settimeout(30)

            logger.info(f"正在连接到目标 {self.target_host}:{self.target_port}...")
            target_socket.connect((self.target_host, self.target_port))
            logger.info(f"成功连接到目标服务器")

            # 设置socket选项
            client_socket.settimeout(30)
            target_socket.settimeout(30)

            # 开始双向转发
            self.forward_data(client_socket, target_socket, client_addr)

        except ConnectionRefusedError:
            logger.error(f"连接目标服务器被拒绝: {self.target_host}:{self.target_port}")
            self.send_error(client_socket, "目标服务器连接被拒绝")
        except socket.timeout:
            logger.error(f"连接目标服务器超时: {self.target_host}:{self.target_port}")
            self.send_error(client_socket, "连接目标服务器超时")
        except Exception as e:
            logger.error(f"处理客户端 {client_addr} 时出错: {e}")
            self.send_error(client_socket, f"内部错误: {str(e)}")
        finally:
            # 关闭连接
            if target_socket:
                try:
                    target_socket.close()
                except:
                    pass
            try:
                client_socket.close()
            except:
                pass

    def forward_data(self, client_socket, target_socket, client_addr):
        """双向转发数据"""
        logger.info(f"开始转发数据: {client_addr[0]}:{client_addr[1]} <-> {self.target_host}:{self.target_port}")

        # 用于同步的event
        stop_event = threading.Event()

        def forward(source, dest, direction_name):
            """单向转发函数"""
            try:
                while not stop_event.is_set():
                    try:
                        data = source.recv(4096)
                        if not data:
                            logger.debug(f"{direction_name} 连接关闭")
                            break
                        dest.send(data)
                    except socket.timeout:
                        continue
                    except (ConnectionResetError, BrokenPipeError):
                        logger.debug(f"{direction_name} 连接重置")
                        break
                    except Exception as e:
                        logger.debug(f"{direction_name} 转发错误: {e}")
                        break
            finally:
                stop_event.set()

        # 创建两个转发线程
        t1 = threading.Thread(
            target=forward,
            args=(client_socket, target_socket, f"{client_addr[0]}:{client_addr[1]} -> 目标")
        )
        t2 = threading.Thread(
            target=forward,
            args=(target_socket, client_socket, f"目标 -> {client_addr[0]}:{client_addr[1]}")
        )

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        # 等待任意一个线程结束（连接关闭）
        try:
            while not stop_event.is_set():
                stop_event.wait(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止转发")
        finally:
            stop_event.set()
            logger.info(f"转发结束: {client_addr[0]}:{client_addr[1]} <-> {self.target_host}:{self.target_port}")

    def send_error(self, client_socket, message):
        """发送错误信息到客户端"""
        try:
            error_msg = f"Port Forwarding Error: {message}\r\n"
            client_socket.send(error_msg.encode())
        except:
            pass

    def stop(self):
        """停止端口转发服务"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
                logger.info("端口转发服务已停止")
            except:
                pass

def check_port_available(host, port):
    """检查端口是否可用"""
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(2)
        test_socket.bind((host, port))
        test_socket.close()
        return True
    except OSError:
        return False

def check_target_reachable(host, port):
    """检查目标地址是否可达"""
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(3)
        result = test_socket.connect_ex((host, port))
        test_socket.close()
        return result == 0
    except:
        return False

def main():
    """主函数"""
    # 配置参数
    LISTEN_HOST = '192.168.137.1'  # 监听地址
    LISTEN_PORT = 10808             # 监听端口
    TARGET_HOST = '127.0.0.1'       # 目标地址
    TARGET_PORT = 10808              # 目标端口

    print("=" * 60)
    print("端口转发工具")
    print("=" * 60)
    print(f"监听地址: {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"转发目标: {TARGET_HOST}:{TARGET_PORT}")
    print("-" * 60)

    # 检查监听端口是否可用
    if not check_port_available(LISTEN_HOST, LISTEN_PORT):
        logger.error(f"监听地址 {LISTEN_HOST}:{LISTEN_PORT} 已被占用")
        logger.info("请关闭其他程序或更换端口")
        sys.exit(1)

    # 检查目标地址是否可达（可选）
    logger.info(f"检查目标服务器 {TARGET_HOST}:{TARGET_PORT} 状态...")
    if check_target_reachable(TARGET_HOST, TARGET_PORT):
        logger.info("✅ 目标服务器可达")
    else:
        logger.warning("⚠️ 目标服务器当前不可达，但转发服务仍会启动")
        logger.warning("   请确保目标服务器稍后会启动")

    print("-" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)

    # 创建并启动转发器
    forwarder = PortForwarder(LISTEN_HOST, LISTEN_PORT, TARGET_HOST, TARGET_PORT)

    try:
        forwarder.start()
    except KeyboardInterrupt:
        logger.info("收到停止信号")
        forwarder.stop()
    except Exception as e:
        logger.error(f"运行时错误: {e}")
        forwarder.stop()

if __name__ == "__main__":
    main()