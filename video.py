from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from utils.utils import *
from models import *
#from util import *
#from darknet import Darknet
#from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from skimage.transform import resize
from tensorflow import keras


#Disnet
import pandas as pd
import argparse

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os

import PIL.Image
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim
       
def resize_img(img, img_size=416):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape
    # Resize and normalize
    input_img = resize(input_img, (img_size,img_size, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float().unsqueeze(0)
    return input_img, img
            
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()



if __name__ == '__main__':
    cfgfile = 'config/yolov3.cfg'
    weightsfile = "weights/yolov3.weights"

    classes = load_classes('data/coco.names')
    cvfont = cv2.FONT_HERSHEY_PLAIN
    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    #cmap = plt.get_cmap('Vega20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    
    CUDA = torch.cuda.is_available()
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    #model.net_info["height"] = args.reso
    #inp_dim = int(model.net_info["height"])
    inp_dim = int(args.reso)
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        print('cam detect cuda is ready')
        
    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor   
         
    model.eval()

    #Disnet
    
    MODEL = "model@1535470106"
    WEIGHTS = "model@1535470106"
    
    # load json and create model
    json_file = open('generated_files/{}.json'.format(MODEL), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json( loaded_model_json )

    # load weights into new model
    loaded_model.load_weights("generated_files/{}.h5".format(WEIGHTS))

    # correct solution
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    #videofile = 'video.avi'
    #cap = cv2.VideoCapture(videofile)
    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    rendering = 1  
    while cap.isOpened():
        

        ret, frame = cap.read()

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
        center  = width/2
        

        prev_time = time.time() 
        if ret:
            #resized_img, img, dim = prep_image(frame, inp_dim)
            resized_img, img = resize_img(frame,inp_dim)
            #print((resized_img.size(),type(resized_img)))
            #cv2.imshow('resized_img',resized_img)    
            input_imgs = Variable(resized_img.type(Tensor))
            #print((resized_img.size(),type(resized_img)))
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                #print(detections)
                alldetections = non_max_suppression(detections, 80, 0.8, 0.4)
                #print('detections get')
                #print((alldetections[0]))
                #print(type(alldetections[0]))
            detection = alldetections[0]
            current_time = time.time()
            inference_time = current_time - prev_time
            #fps = int(1/(inference_time))
            fps =  cap.get(cv2.CAP_PROP_FPS)
            print('current fps is : %d'%fps)
    
            if(rendering):
                kitti_img_size = 416
                # The amount of padding that was added
                #pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
                #pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
                pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))
                # Image height and width after padding is removed
                unpad_h = kitti_img_size - pad_y
                unpad_w = kitti_img_size - pad_x
        
                # Draw bounding boxes and labels of detections
                if detection is not None:
                    #print(img.shape)
                    unique_labels = detection[:, -1].cpu().unique()
                    n_cls_preds = min(len(unique_labels),20)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        loc = None
                        centroid = (x1+x2)/2
                    
                        if centroid< (center - center/2 ): loc = "left"
                        elif centroid < center: loc = "center"
                        elif centroid < (center + center/3 ): loc="right"
                        cls_pred = min(cls_pred,80)     
                        #print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                        # Rescale coordinates to original dimensions
                        box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                        box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]) )
                        y1b = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                        x1b = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
                        x2b = int(x1b + box_w)
                        y2b = int(y1b + box_h)
                        #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # get data
                        
                        X_test = [[x1, y1, x2, y2]]

                        # standardized data
                        scaler = StandardScaler()
                        
                        X_test = scaler.fit_transform(X_test)

                        # evaluate loaded model on test data
                        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
                        y_pred = loaded_model.predict(X_test)

                        #y_pred = y_pred*(-1)
                        #scale = np.round(y_pred*10)%10
                        #y_pred = y_pred+1000*(scale)
                        #y_pred = y_pred//100

                        # create empty table with 12 fields
                        trainPredict_dataset_like = np.zeros(shape=(len(y_pred), 4) )
                        # put the predicted values in the right field
                        trainPredict_dataset_like[:,0] = y_pred[:,0]
                        # inverse transform and then select the right field
                        y_pred = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
                        #y_pred = scaler.inverse_transform(y_pred)
                        y_pred = (y_pred - 25)/100
                        if loc =="left": y_pred+=1.5
                        elif loc=="right": y_pred-=1
                        # scale up predictions to original values
                        #y_pred = scaler.inverse_transform(y_pred)
                        print(y_pred)
                    
                        #print(color)
                        #cv2.line(img,(int(x1), int(y1-5)),(int(x2), int(y1-5)),(255,255,255),14)
                        #cv2.putText(img,classes[int(cls_pred)],(int(x1), int(y1), int(y_pred)), cvfont, 1.5, (color[0]*255,color[1]*255,color[2]*255),2)

                        cv2.putText(img, classes[int(cls_pred)], (int(x1), int(y1)), cvfont, 1.5, (255, 0, 0),2)
                        cv2.rectangle(img, (int(x1b),int(y1b)), (int(x2b),int(y2b)), (255, 0, 0), 1)

                        
                        string = "{} m".format(y_pred)
                        cv2.putText(img, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        print(classes[int(cls_pred)] , string , loc)
                        
                cv2.imshow('frame', img)
                # free buffer
                #cv2.imshow('kitti detecting window',plt)
                #plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
                key = cv2.waitKey(1)  
                if key & 0xFF == ord('q'):
                    break  
                
        #no ret berak         
        else:
            break
    

    
    

