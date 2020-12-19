import sys
import socket
import logging
from PIL import Image
import io
import time

TAILCODE = b'$@##'

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect(('127.0.0.1', 5678))
        except:
            print('connection fail, try again')
            time.sleep(1)
        print('connection established')
        break
        
    while True:
        data = open('data/demo_images/00000001.jpg', 'rb').read()
        s.send(data+TAILCODE)
        time.sleep(0.02)