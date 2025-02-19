import numpy as np
import random
import torch
#import torch.nn as nn
import os, sys
import cv2
from copy import deepcopy, copy
import yaml
from pathlib import Path

yoloGitPath = "yolov7/"
sys.path.append('yolov7')

from utils.datasets import letterbox
from utils.general import non_max_suppression, xyxy2xywh, scale_coords
from utils.plots import plot_one_box

from torchview import draw_graph

from models.yolo import Model

torch.set_printoptions(profile="full") 

os.environ['CUDA_VISIBLE_DEVICES'] = ""
device ='cpu'

image_path = yoloGitPath + "inference/images/horses.jpg"

result_path = "split_yolov7_model/"

first_half_path = result_path + "yolov7_first_half.pt"
second_half_path = result_path + "yolov7_second_half.pt"


# function for preparing the test image
def prepare_test_image(im0):
    print("prepare test image")
    
    # Padded resize
    _img = letterbox(copy(im0))[0]

    # Convert
    _img = _img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    _img = np.ascontiguousarray(_img)

    _img = torch.from_numpy(_img)#.to(device)
    _img = _img.float()  # uint8 to fp16/32
    _img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if _img.ndimension() == 3:
        _img = _img.unsqueeze(0)

    return _img

# main function
if __name__ == '__main__':

    # load and prepare the test image image
    img0 = cv2.imread(image_path)  # BGR
    img = prepare_test_image(img0)
    
    print("load the splited model")
    firsModel = torch.load(first_half_path, weights_only=False, map_location = device).float().eval()
    secModel = torch.load(second_half_path, weights_only=False, map_location = device).float().eval()

    
    print("run inference")
    mid_pred = firsModel(img)
    pred = secModel(mid_pred)[0]
    pred_nms = non_max_suppression(pred)

    
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    print("Split model:",len(pred_nms[-1]), "detection/s in this image:")
    for i, det in enumerate(pred_nms):

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            print("Class:", int(cls)," conf:", float(conf), "xywh:", xywh)
            label = f'{int(cls)} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=(0,0,255), line_thickness=1)


    # save the test image with all boxes drawn 
    cv2.imwrite(result_path + "test_2.png", img0)

    # create and save the graph of the original and both parts of the split model 
    #draw_graph(firsModel, input_data=img, expand_nested=True, save_graph=True, filename= result_path + "yolov7_first_half_graph")
    #draw_graph(secModel, input_data=mid_pred, expand_nested=True, save_graph=True, filename= result_path + "yolov7_second_half_graph")

    #print(pred_original)
    #print(pred)





