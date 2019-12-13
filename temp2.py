from PIL import Image
import cv2
import urllib.request
import numpy as np
# http://192.168.8.101/
stream = urllib.request.urlopen('http://192.168.8.101:81/stream.mjpg')
bytes = bytes()
while True:
    bytes += stream.read(2048)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        # try:
            # print(bytes)
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        open_cv_image = np.array(i)
        i = i[:,:,::-1].copy()

        cv2.imshow('i',i)
        # cv2_im_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        # pil_im = Image.fromarray(cv2_im_rgb)
        # print(type(pil_im))

        # cv2.imshow('pil_im', pil_im)
        # print(i.shape)
        # print(type(i))
        #     pass
        # except:
            # print("E")        
        
        
    if cv2.waitKey(1) == 27:
        exit(0)