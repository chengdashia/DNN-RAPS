"""
 配置相关
"""
server_list = [
    {
        "ip": "192.168.215.129",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001
        }
    },
    {
        "ip": "192.168.215.130",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001
        }
    },
    {
        "ip": "192.168.215.131",
        "username": "root",
        "password": "123456",
        "application": {
            "VGG5": 9001
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
