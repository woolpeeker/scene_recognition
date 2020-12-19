import sys
import socket
import logging
from PIL import Image
import io
from easydict import EasyDict as edict
import onnx
import onnxruntime
import numpy as np

CFG = edict()
CFG.onnx = 'checkpoints/resnet18/model.onnx'

def get_logger():
    logger = logging.getLogger()    # initialize logging class
    format = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(format)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    return logger

class Detector:
    def __init__(self, onnx):
        self.cate_names = [
            '/a/airfield', '/a/art_school', '/b/bridge', '/c/crosswalk', '/d/downtown',
            '/a/army_base', '/f/fire_station', '/h/highway', '/g/gas_station', '/c/church/outdoor'
        ]
        self.onnx = onnx
        self.onnx_sess = onnxruntime.InferenceSession(onnx)
        input_names = [node.name for node in self.onnx_sess.get_inputs()]
        output_names = [node.name for node in self.onnx_sess.get_outputs()]
        logger.info(f'input_names: {input_names}')    
        logger.info(f'output_names: {output_names}')
        assert len(input_names) == 1
        assert len(output_names) == 1
        self.inp_name = input_names[0]
        self.out_name = output_names[0]
    
    def detect(self, image):
        img = image.convert('RGB').resize((224,224))
        img = np.array(img).astype(np.float32)
        image_tensor = img.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        out = self.onnx_sess.run([self.out_name], input_feed={self.inp_name: image_tensor})
        scores = out[0]
        class_idx = np.argmax(scores)
        logger.info(f"{self.cate_names[class_idx]}")

if __name__ == '__main__':
    logger = get_logger()
    detector = Detector(CFG.onnx)

    ADDR = ('', 5678) # host, port
    IMG_FORMAT = 'JPG'
    TAILCODE = b'$@##'

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen(1)

    sock, addr = server.accept()
    logger.info('Accept new connection from %s:%s...' % addr)
    # deal with client
    all_data = b''
    while True:
        data = sock.recv(1024)
        all_data += data
        idx = all_data.find(TAILCODE)
        if idx != -1:
            logger.debug('received end of image')
            imgByteArr = io.BytesIO(all_data[:-len(TAILCODE)])
            img = Image.open(imgByteArr)
            detector.detect(img)
