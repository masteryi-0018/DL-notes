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
    代理请求处理器 - 支持HTTP和HTTPS的TCP转发代理
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

            # 解析请求
            try:
                lines = request.split(b'\r\n')
                if lines and len(lines) > 0:
                    first_line = lines[0].decode('utf-8', errors='ignore')
                    parts = first_line.split()

                    if len(parts) >= 2:
                        method = parts[0]

                        # 处理CONNECT方法（HTTPS隧道）
                        if method == 'CONNECT':
                            self.handle_connect(parts[1])
                        else:
                            # 处理HTTP请求
                            self.handle_http(request, lines)
                    else:
                        self.request.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                else:
                    self.request.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")

            except Exception as e:
                logger.error(f"解析请求时出错: {e}")
                try:
                    self.request.send(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                except:
                    pass

        except Exception as e:
            logger.error(f"处理连接时出错: {e}")

    def handle_connect(self, target):
        """
        处理CONNECT方法（HTTPS隧道）
        """
        try:
            # 解析目标主机和端口
            if ':' in target:
                host, port = target.split(':')
                port = int(port)
            else:
                host = target
                port = 443  # HTTPS默认端口

            logger.info(f"CONNECT请求: {host}:{port}")

            # 连接到目标服务器
            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_socket.connect((host, port))

            # 发送200 Connection Established响应
            self.request.send(b"HTTP/1.1 200 Connection Established\r\n\r\n")

            # 设置超时
            self.request.settimeout(30)
            target_socket.settimeout(30)

            # 开始双向转发数据
            self.forward_data(self.request, target_socket)

        except Exception as e:
            logger.error(f"处理CONNECT请求时出错: {e}")
            try:
                self.request.send(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            except:
                pass

    def handle_http(self, request, lines):
        """
        处理HTTP请求
        """
        try:
            # 解析目标地址和端口
            first_line = lines[0].decode('utf-8', errors='ignore')
            parts = first_line.split()
            url = parts[1]

            # 提取主机名
            if url.startswith('http://'):
                # HTTP请求
                host_start = url.find('://') + 7
                host_end = url.find('/', host_start)
                if host_end == -1:
                    host_end = len(url)
                host_part = url[host_start:host_end]

                # 检查是否有端口
                if ':' in host_part:
                    host, port = host_part.split(':')
                    port = int(port)
                else:
                    host = host_part
                    port = 80

                logger.info(f"HTTP请求: {host}:{port}")

                # 连接到目标服务器
                target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                target_socket.connect((host, port))

                # 转发请求
                target_socket.send(request)

                # 接收响应并转发
                self.forward_response(target_socket, self.request)

                target_socket.close()
            else:
                # 可能是相对路径的HTTP请求
                # 尝试从Host头获取主机信息
                host = None
                port = 80

                for line in lines[1:]:
                    if line.startswith(b'Host:'):
                        host_line = line.decode('utf-8', errors='ignore')
                        host_part = host_line[5:].strip()
                        if ':' in host_part:
                            host, port_str = host_part.split(':')
                            port = int(port_str)
                        else:
                            host = host_part
                        break

                if host:
                    logger.info(f"HTTP请求(从Host头): {host}:{port}")

                    # 连接到目标服务器
                    target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    target_socket.connect((host, port))

                    # 转发请求
                    target_socket.send(request)

                    # 接收响应并转发
                    self.forward_response(target_socket, self.request)

                    target_socket.close()
                else:
                    logger.warning("无法解析目标主机")
                    self.request.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")

        except Exception as e:
            logger.error(f"处理HTTP请求时出错: {e}")
            try:
                self.request.send(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
            except:
                pass

    def forward_data(self, client_socket, target_socket):
        """
        双向转发数据
        """
        try:
            # 创建两个线程分别处理双向数据转发
            threads = []

            # 客户端到目标
            t1 = threading.Thread(target=self.forward_one_way,
                                 args=(client_socket, target_socket, "客户端->目标"))
            t1.daemon = True
            threads.append(t1)

            # 目标到客户端
            t2 = threading.Thread(target=self.forward_one_way,
                                 args=(target_socket, client_socket, "目标->客户端"))
            t2.daemon = True
            threads.append(t2)

            # 启动线程
            for t in threads:
                t.start()

            # 等待其中一个线程结束（连接关闭）
            for t in threads:
                t.join()

        except Exception as e:
            logger.error(f"数据转发时出错: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            try:
                target_socket.close()
            except:
                pass

    def forward_one_way(self, source, destination, direction):
        """
        单向数据转发
        """
        try:
            while True:
                data = source.recv(4096)
                if not data:
                    break
                destination.send(data)
        except Exception as e:
            logger.debug(f"{direction} 转发结束: {e}")

    def forward_response(self, source, destination):
        """
        转发响应数据
        """
        try:
            while True:
                data = source.recv(4096)
                if not data:
                    break
                destination.send(data)
        except Exception as e:
            logger.debug(f"响应转发结束: {e}")

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
        logger.info("现在支持HTTP和HTTPS连接")
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