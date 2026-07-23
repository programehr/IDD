import os
import pickle

"""
Defines a class that stores an array in a binary file and can append new elements to it.
"""

class FileArray:
    def __init__(self, path):
        if not os.path.exists(path):
            with open(path, 'wb+') as f:
                pass
        self.path = path
        self.index = 0
        self.position = 0

    def append(self, element):
        byte_array = pickle.dumps(element)
        length = format(len(byte_array), '032b')
        length = bytes(length, 'ascii')
        byte_array = length + byte_array
        with open(self.path, 'ab') as f:
            f.seek(0, os.SEEK_END)
            f.write(byte_array)

    def read(self):
        array = []
        with open(self.path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)
            while True:
                length = f.read(32)
                length = int(length, 2)
                bytes_array = f.read(length)
                element = pickle.loads(bytes_array)
                array.append(element)
                current = f.tell()
                if current >= size:
                    break
        return array

    def __iter__(self):
        return FileArrayIterator(self.path)

class FileArrayIterator:
    def __init__(self, path):
        self.path = path
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if self.position >= size:
                raise StopIteration
            f.seek(self.position)
            length = f.read(32)
            length = int(length, 2)
            bytes_array = f.read(length)
            element = pickle.loads(bytes_array)
            self.position = f.tell()
        return element


if __name__ == '__main__':
    t = FileArray('test.bin')
    t.append([1, 2, 3])
    t.append([4, 5, 6, 7, 8])
    t.append(9)
    for x in t:
        print(x)