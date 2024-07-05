import socket
import threading
from network_utils import send_data, receive_data

node_layer_indices = {
    'task1': {'client1': [0, 1], 'client2': [2, 3, 4, 5, 6]},
    'task2': {'client2': [0, 1, 2], 'client3': [3, 4, 5, 6]},
    'task3': {'client3': [0, 1, 2, 3], 'client1': [4, 5, 6]}
}

initial_data = [1, 2, 3, 4, 5, 6, 7]
clients = {}
task_results = {}
client_status = {client: True for client in set(client for task in node_layer_indices.values() for client in task)}
task_status = {task: False for task in node_layer_indices}
lock = threading.Lock()
condition = threading.Condition(lock)


def handle_client(conn, client_name, task_name, data_indices, target_indices):
    data_to_send = (data_indices, target_indices)
    send_data(conn, data_to_send)
    message = receive_data(conn)
    if not message:
        return None, None, None

    process_time, processed_data, processed_target = message
    print(f"{client_name} completed part of {task_name} in {process_time} seconds")

    return processed_data, processed_target, process_time


def process_task(task_name, client_name, data_indices, target_indices):
    processed_data, processed_target, _ = handle_client(clients[client_name], client_name, task_name, data_indices,
                                                        target_indices)

    if processed_data is not None:
        with lock:
            task_results.setdefault(task_name, []).append((processed_data, processed_target))
            client_status[client_name] = True
            condition.notify_all()
    else:
        print(f"{client_name} received empty data. Closing connection.")
        clients[client_name].close()


def manage_tasks():
    for task_name, clients_indices in node_layer_indices.items():
        first_client = list(clients_indices.keys())[0]
        data_indices = clients_indices[first_client]
        target_indices = data_indices

        with lock:
            client_status[first_client] = False

        threading.Thread(target=process_task, args=(task_name, first_client, data_indices, target_indices)).start()

    while True:
        with lock:
            all_tasks_done = True
            for task_name, clients_indices in node_layer_indices.items():
                if task_name not in task_results or len(task_results[task_name]) < 2:
                    all_tasks_done = False
                    second_client = list(clients_indices.keys())[1]
                    if client_status[second_client]:
                        data_indices = clients_indices[second_client]
                        target_indices = data_indices
                        client_status[second_client] = False
                        threading.Thread(target=process_task,
                                         args=(task_name, second_client, data_indices, target_indices)).start()

            if all_tasks_done:
                break

            condition.wait()


def accept_connections(server_socket):
    while len(clients) < 3:  # Assuming there are three clients to connect
        conn, addr = server_socket.accept()
        client_name = receive_data(conn)
        if client_name:
            clients[client_name] = conn
            print(f"{client_name} connected from {addr}")

    manage_tasks()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 12345))
    server_socket.listen()
    print("Server is listening for connections...")
    accept_connections(server_socket)
    server_socket.close()


if __name__ == "__main__":
    main()
