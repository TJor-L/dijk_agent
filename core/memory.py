from typing import List
from .message import Message

class ShortTermMemory:
    def __init__(self, k: int = 20):
        self.buffer: List[Message] = []
        self.k = k

    def append(self, m: Message):
        self.buffer.append(m)
        if len(self.buffer) > 2000:
            self.buffer = self.buffer[-2000:]

    def window(self) -> List[Message]:
        return self.buffer[-self.k:]
