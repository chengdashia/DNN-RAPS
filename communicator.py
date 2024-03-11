# Communicator Object
import pickle
import struct
import socket

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Communicator(object):
	def __init__(self, host_ip, host_port):
		self.sock = socket.socket()
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
		self.sock.bind((host_ip,host_port))  # Bind the socket to a specific address and port
		print("host_ip:{}, host_port:{}".format(host_ip,host_port))

	def wait_for_connection(self):
		self.sock.listen()  # Enable the server to accept connections
		node_sock, node_address = self.sock.accept()  # Wait for a client to connect
		return node_sock, node_address

	def send_msg(self, sock, msg):
		msg_pickle = pickle.dumps(msg)
		sock.sendall(struct.pack(">I", len(msg_pickle)))
		sock.sendall(msg_pickle)
		logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	def recv_msg(self, sock, expect_msg_type=None):
		msg_len = struct.unpack(">I", sock.recv(4))[0]
		msg = sock.recv(msg_len, socket.MSG_WAITALL)
		msg = pickle.loads(msg)
		logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

		# TODO:修改msg信息格式
		#if expect_msg_type is not None:
		#	if msg[0] == 'Finish':
		#		return msg
		#	elif msg[0] != expect_msg_type:
		#		raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
		return msg
	# def recv_msg(self, sock, expect_msg_type=None):
	# 	# 使用辅助函数接收完整的4个字节数据
	# 	raw_msglen = self.recvall(sock, 4)
	# 	if not raw_msglen:
	# 		return None
	# 	msg_len = struct.unpack(">I", raw_msglen)[0]
	#
	# 	# 接收消息主体
	# 	msg = self.recvall(sock, msg_len)
	# 	if not msg:
	# 		return None
	#
	# 	msg = pickle.loads(msg)
	# 	logger.debug(msg[0] + ' received from ' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
	#
	# 	# TODO:修改msg信息格式
	# 	# if expect_msg_type is not None:
	# 	#     if msg[0] == 'Finish':
	# 	#         return msg
	# 	#     elif msg[0] != expect_msg_type:
	# 	#         raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
	#
	# 	return msg
	#
	# def recvall(self, sock, n):
	# 	# 辅助函数，用于接收完整的数据
	# 	data = bytearray()
	# 	while len(data) < n:
	# 		packet = sock.recv(n - len(data))
	# 		if not packet:
	# 			return None
	# 		data.extend(packet)
	# 	return data