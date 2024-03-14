"""
 用于通信的json数据的，python对象
"""
import json


class Message:
    def __init__(self, message_type, data, sender=None, target=None):
        self.type = message_type
        self.data = data
        self.sender = sender
        self.target = target

    def to_json(self):
        message_dict = {
            'type': self.type,
            'data': self.data
        }
        if self.sender:
            message_dict['sender'] = self.sender
        if self.target:
            message_dict['target'] = self.target
        return json.dumps(message_dict)

    @staticmethod
    def from_json(json_data):
        message_dict = json.loads(json_data)
        message_type = message_dict['type']
        data = message_dict['data']
        sender = message_dict.get('sender', None)
        target = message_dict.get('target', None)
        return Message(message_type, data, sender, target)