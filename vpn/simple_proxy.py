#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import threading
import socketserver
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProxyHandler(socketserver.StreamRequestHandler):
    """
    代理请求处理器 - 简单的TCP转发代理
    """

    def handle(self):
        """
        处理客户端连接
        """
        try:
            # 解析客户端请求
            request = self.request.recv(4096)
            if not request:
                return

            logger.info(f"收到来自 {self.client_address} 的请求")

            # 解析目标地址和端口（从HTTP请求中）
            try:
                # 尝试解析HTTP请求获取目标主机
                lines = request.split(b'\r\n')
                if lines and len(lines) > 0:
                    first_line = lines[0].decode('utf-8', errors='ignore')
                    parts = first_line.split()
                    if len(parts) >= 2:
                        url = parts[1]
                        if url.startswith(b'http://'.decode() if isinstance(url, str) else 'http://'):
                            # 提取主机名
                            host_start = url.find('://') + 3
                            host_end = url.find('/', host_start)
                            if host_end == -1:
                                host_end = len(url)
                            host = url[host_start:host_end]

                            # 检查是否有端口
                            if ':' in host:
                                target_host, target_port = host.split(':')
                                target_port = int(target_port)
                            else:
                                target_host = host
                                target_port = 80

                            logger.info(f"目标服务器: {target_host}:{target_port}")

                            # 连接到目标服务器
                            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            target_socket.connect((target_host, target_port))

                            # 转发请求
                            target_socket.send(request)

                            # 设置超时
                            target_socket.settimeout(10)

                            # 接收响应并转发回客户端
                            while True:
                                try:
                                    response = target_socket.recv(4096)
                                    if not response:
                                        break
                                    self.request.send(response)
                                except socket.timeout:
                                    break
                                except Exception as e:
                                    logger.error(f"接收响应时出错: {e}")
                                    break

                            target_socket.close()

                        else:
                            # 非HTTP请求或无法解析
                            logger.warning(f"非HTTP请求或无法解析: {first_line[:50]}...")
                            self.request.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                    else:
                        self.request.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                else:
                    self.request.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")

            except Exception as e:
                logger.error(f"处理请求时出错: {e}")
                try:
                    self.request.send(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                except:
                    pass

        except Exception as e:
            logger.error(f"处理连接时出错: {e}")

class ThreadedProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """多线程代理服务器"""
    allow_reuse_address = True
    daemon_threads = True

def main():
    """
    主函数
    """
    # 服务器配置
    HOST = '127.0.0.1'
    PORT = 10808

    try:
        # 创建代理服务器
        server = ThreadedProxyServer((HOST, PORT), ProxyHandler)
        logger.info(f"代理服务器启动在 {HOST}:{PORT}")
        logger.info("请在系统代理设置中配置: 127.0.0.1:10808")
        logger.info("注意: 这是一个简单的转发代理，不会修改任何数据")
        logger.info("按 Ctrl+C 停止服务器")

        # 启动服务器
        server.serve_forever()

    except KeyboardInterrupt:
        logger.info("正在停止代理服务器...")
        server.shutdown()
        server.server_close()
        logger.info("代理服务器已停止")
        sys.exit(0)

    except Exception as e:
        logger.error(f"启动服务器时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()