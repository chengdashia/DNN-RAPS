"""
 配置相关
"""
local_server_list = [
    {
        "name": "client1",
        "ip": "127.0.0.1",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001
        }
    },
    {
        "name": "client2",
        "ip": "127.0.0.1",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9002
        }
    },
    {
        "name": "client3",
        "ip": "127.0.0.1",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9003
        }
    }
]
server_list = [
    {
        "ip": "192.168.215.129",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001,
            "VGG6": 9002
        }
    },
    {
        "ip": "192.168.215.130",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001,
            "VGG6": 9002
        }
    },
    {
        "ip": "192.168.215.131",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001,
            "VGG6": 9002
        }
    }
    # 添加更多服务器
]
CLIENTS_CONFIG = {server["ip"]: i for i, server in enumerate(server_list)}
CLIENTS_LIST = [server["ip"] for server in server_list]
CLIENTS_NUMBERS = len(CLIENTS_LIST)

# data length
N = 10000
# Batch size
B = 256
