from config import server_list,CLIENTS_CONFIG,CLIENTS_LIST
from utils.resource_utilization import get_all_server_info
if __name__ == '__main__':
    infos = get_all_server_info(server_list)
    print(infos)
    print(CLIENTS_CONFIG)
    print(CLIENTS_LIST)