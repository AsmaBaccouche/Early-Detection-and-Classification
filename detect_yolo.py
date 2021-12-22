# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:01:56 2020

@author: Asma Baccouche
"""

import os, glob
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3_model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3_utils import letterbox_image
from keras.utils import multi_gpu_model

#n = 'mass'
#f = 'ep063-loss13.737-val_loss14.094'
#n = 'calc'
#f = 'ep057-loss13.444-val_loss13.054'
n = 'arch'
f = 'ep054-loss13.318-val_loss13.017'
#n = 'all'
#f = 'ep063-loss10.300-val_loss10.247'

#csvname = 'prior_'+n+'_cyclegan' 
csvname = n+'3'
foldername = csvname
try:
    os.mkdir(foldername)
except:
    print("Folder already exists!")

images = glob.glob('D:/Yufeng_Data/Current_augmented/*.jpg')
#images = glob.glob('D:/Yufeng_Data/Prior_augmented/*.jpg')
#images = glob.glob('D:/Yufeng_Data/Prior_to_Current_cyclegan/*.jpg')
#images = glob.glob('D:/Yufeng_Data/Prior_to_Current_pix2pix/*.jpg')
#images = [file for file in images if n.upper() in file]
#images = [file for file in images if 'MASS' not in file and 'ARCH' not in file and 'CALC' not in file]
#images = [file for file in images if 'ARCH' in file or 'CALC' in file]

#images = glob.glob('non_calc_mix/*.jpg')
#images = [file[:-9]+'.png' for file in images]


class YOLO(object):

    name = 'yufeng_'+n+'_annotation'
    _defaults = {
        "model_path": 'logs_'+name+'/'+f+'.h5',
        "anchors_path": name+'_anchor.txt',
        "classes_path": n+'.txt',
        "score" : 0.35,
        "iou" : 0.45,
        "model_image_size" : (448, 448),
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        if len(self.class_names) == 1:
            self.colors = [(0,255,0)]
        else:
            if len(self.class_names) == 2:
                self.colors = [(255, 255, 0), (0, 255, 0)]
            else:
                self.colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255)]
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='YOLO files/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        l=[]
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin)],fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            l.append(label)
        return image, l

    def close_session(self):
        self.sess.close()
 

#import time

#s = time.time()

yolo = YOLO()
names = []
predclasses = []
for image_path in images:
    image = Image.open(image_path)
    #array = np.uint8(np.array(image) / 256)
    #image = Image.fromarray(array) 
    image = image.convert('RGB')
    pred, label = yolo.detect_image(image)
    name = image_path.split('\\')[-1][:-4]
    names.append(name)
    predclasses.append(label)
    image.save(os.path.join(foldername,name+"_pred.png"))    
    
#t = time.time()

#print((t-s)/len(images))

    
benchmark = pd.DataFrame({'Case': names,'Predicted class': predclasses})
benchmark.to_csv(os.path.join(foldername,csvname + ".csv"))



import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('D:\Yufeng_Data\Prior_augmented/0_PATIENT049LNLLMLO.jpg')
p = 'D:\PhD progress\Yufeng project\with_prior_with_current/0_PATIENT049LNLLMLO_pred.png'
img2 = cv2.imread(p)
img1 = cv2.resize(img1, (256,256))

plt.imshow(img1)
plt.imshow(img2)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

lower_yellow = np.array([25,50,70])
upper_yellow = np.array([35,255,255])

lower_green = np.array([36,25,25])
upper_green = np.array([70,255,255])

hsv_img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
masking = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
_,thresh = cv2.threshold(masking,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]

areas = [cv2.contourArea(cnt) for cnt in contours]
cnt = contours[np.argmax(areas)]
cv2.drawContours(img1, [cnt], 0, (255, 0, 0))
plt.imshow(img1)

cv2.imwrite(p, img1)