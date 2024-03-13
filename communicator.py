# Communicator Object
import json
import struct
import socket
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
	def __init__(self, host_ip, host_port):
		# 创建socket对象
		self.sock = socket.socket()
		# 设置socket以便地址重用
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		# 绑定socket到特定IP和端口
		self.sock.bind((host_ip, host_port))
		# 用于存储对端的地址信息
		self.peername = None
		logging.info("host_ip:{}, host_port:{}".format(host_ip, host_port))

	def wait_for_connection(self):
		# 监听等待客户端连接
		self.sock.listen()
		# 等待客户端连接
		node_sock, node_address = self.sock.accept()
		return node_sock, node_address

	def send_message(self, sock, msg):
		# 将消息序列化
		msg_pickle = json.dumps(msg)
		# 首先发送json字符串的长度
		sock.sendall(struct.pack(">I", len(msg_pickle)))
		# 然后发送JSON字符串本身
		sock.sendall(msg_pickle.encode('utf-8'))
		# 记录发送日志
		logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	def receive_message(self, sock, expect_msg_type=None):
		# 收到消息长度信息后，接收相应长度的消息内容
		msg_len = struct.unpack(">I", sock.recv(4))[0]
		# 接收完整消息
		msg = sock.recv(msg_len, socket.MSG_WAITALL).decode('utf-8')
		# 反序列化消息
		msg = json.loads(msg)
		# 记录接收日志

		logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

		# 根据预期的消息类型进行校验
		if expect_msg_type is not None:
			if msg[0] == 'Finish':
				return msg
			elif msg[0] != expect_msg_type:
				raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
		return msg
