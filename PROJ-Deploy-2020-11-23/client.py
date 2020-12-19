import sys
import socket
import logging
from PIL import Image
import io
import time

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data = open(sys.argv[1], 'rb').read()
    while True:
        try:
            s.connect(('127.0.0.1', 8002))
        except:
            print('connection fail, try again')
            time.sleep(1)
        print('connection established')
        break
    s.send(data)
