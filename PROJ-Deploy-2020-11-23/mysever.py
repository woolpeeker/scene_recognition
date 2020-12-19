import socket
import time
import cv2
import numpy
import numpy as np
import sys
import logging
from PIL import Image
import io
from easydict import EasyDict as edict
from pathlib import Path
import onnx
import onnxruntime
from label_cfg import LABEL_CFG, INOUT_CFG


def get_logger():
    logger = logging.getLogger()    # initialize logging class
    format = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(format)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    return logger


def recvall(sock, count):
    buf = b''#buf是一个byte类型
    while count:
    #接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

class Detector:
    def __init__(self, inout_cfg, label_cfg, onnx):
        self.inout_cfg = inout_cfg
        self.label_cfg = label_cfg
        self.onnx = onnx
        self.onnx_sess = onnxruntime.InferenceSession(onnx)
        self.inp_names = [node.name for node in self.onnx_sess.get_inputs()]
        self.out_names = [node.name for node in self.onnx_sess.get_outputs()]
        logger.info(f'input_names: {self.inp_names}')    
        logger.info(f'output_names: {self.out_names}')
        assert len(self.inp_names) == 1
        assert len(self.out_names) == 2
        assert self.out_names[0] == 'inout_prob' and self.out_names[1] == 'label_prob'
    
    def detect(self, image):
        img = image.convert('RGB').resize((224,224))
        img = numpy.array(img).astype(numpy.float32) / 255
        image_tensor = img.transpose(2, 0, 1)
        image_tensor = image_tensor[numpy.newaxis, :]
        out = self.onnx_sess.run(self.out_names, input_feed={self.inp_names[0]: image_tensor})
        print(out)
        output = self.postprocess(out)
        logger.info(str(output))
        return output

    def postprocess(self, data):
        idx_lst = np.argmax(data[0], axis=-1).tolist()
        inout_keys = list(self.inout_cfg.keys())
        inout_names = [inout_keys[i] for i in idx_lst]
        idx_lst = np.argmax(data[1], axis=-1).tolist()
        label_keys = list(self.label_cfg.keys())
        label_names = [label_keys[i] for i in idx_lst]
        return ';'.join([x+','+y for x,y in zip(inout_names, label_names)])

def test(img_file):
    detector = Detector(INOUT_CFG, LABEL_CFG, './resnet18-inout.onnx')
    if img_file.endswith('.txt'):
        img_files = np.loadtxt(img_file, dtype=str).tolist()
        root = img_files[0]
        img_files = [Path(root) / x for x in img_files[1:]]
    else:
        img_files = [img_file]
    fp = open('preds.txt', 'w')
    for img_file in img_files:
        img = Image.open(str(img_file))
        output = detector.detect(img)
        fp.write(str(img_file) + ':' + str(output)+ '\n')

def start_server():
    detector = Detector(INOUT_CFG, LABEL_CFG, './resnet18-inout.onnx')
    #IP地址'0.0.0.0'为等待客户端连接
    address = ('0.0.0.0', 8002)
    #socket.AF_INET：服务器之间网络通信 
    #socket.SOCK_STREAM：流式socket , for TCP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #将套接字绑定到地址, 在AF_INET下,以元组（host,port）的形式表示地址.
    s.bind(address)
    #开始监听TCP传入连接。参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
    s.listen(1)
    #接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
    #没有连接则等待有连接
    while 1:
        conn, addr = s.accept()
        while 1:
            #终止本次握手
            length = recvall(conn,16)
            if int(length)==0:
                break
            #reciceve the picture
            if int(length)>100:
                #do cliassification
                stringData = recvall(conn, int(length))#根据获得的文件长度，获取图片文件
                imgByteArr = io.BytesIO(stringData)
                img = Image.open(imgByteArr)
                name=detector.detect(img)
                #show the image frame
                #data = numpy.frombuffer(stringData, numpy.uint8)#将获取到的字符流数据转换成1维数组
                #decimg=cv2.imdecode(data,cv2.IMREAD_COLOR)#将数组解码成图像
                #cv2.imshow('SERVER',decimg)#显示图像
                #把场景识别结果回传
                conn.send(bytes(str(name),encoding='utf-8'))
                print('message send: ', str(name))
            #接受信息
            if 1<int(length)<100:
                #recieve the message
                message= recvall(conn, int(length))
                print(str(message,encoding='utf-8'))
                    
    s.close()
    cv2.destroyAllWindows()


        

if __name__ == '__main__':
    logger = get_logger()
    if len(sys.argv) > 1:
        test(sys.argv[1])
    else:
        start_server()
    
