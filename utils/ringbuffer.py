import numpy as np


class RingBuffer(object):

    def __init__(self, item_len, capacity):
        self.item_len = item_len
        self.capacity = capacity
        self.buffer = np.zeros((item_len, capacity))
        self.n_items = 0
        self.idx = 0

    def add(self, item):
        self.buffer[:, self.idx] = item
        self.idx = (self.idx + 1) % self.capacity
        self.n_items = min(self.n_items + 1, self.capacity)

    def get(self, n_items):
        if n_items > self.n_items:
            return None
        else:
            if n_items <= self.idx:
                return self.buffer[:, self.idx-n_items:self.idx]
            else:
                return np.hstack([self.buffer[:, -(n_items-self.idx):], self.buffer[:, :self.idx]])

    def get_all(self):
        if self.n_items < self.capacity:
            return self.buffer[:, :self.idx]
        else:
            return np.hstack([self.buffer[:, self.idx:], self.buffer[:, :self.idx]])

    def is_full(self):
        return self.n_items == self.capacity

    def num_items(self):
        return self.n_items
