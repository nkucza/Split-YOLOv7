import numpy as np
import random
import torch
#import torch.nn as nn
import os, sys, shutil
import cv2
from copy import deepcopy, copy
import yaml
import math

from pathlib import Path

yoloGitPath = "yolov7/"
sys.path.append('yolov7')

from utils.datasets import letterbox
from utils.general import non_max_suppression, xyxy2xywh, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import initialize_weights
from utils.google_utils import attempt_download

from torchview import draw_graph

from models.yolo import Model

torch.set_printoptions(profile="full") 

os.environ['CUDA_VISIBLE_DEVICES'] = ""
device ='cpu'

weights = "yolov7.pt"

file = Path(str(weights).strip().replace("'", '').lower())

if not file.exists():
    print("need to download the weights")
    os.chdir(yoloGitPath)
    attempt_download(weights)
    os.chdir("..")
    shutil.move(yoloGitPath + weights, ".")


image_path = yoloGitPath + "inference/images/horses.jpg"

result_path = "split_yolov7_model/"

Path(result_path).mkdir(parents=True, exist_ok=True)

first_half_yaml_path = result_path + "yolov7_first_half.yaml"
first_half_state_dict_path = result_path + "yolov7_first_half_state_dict.pt"
first_half_full_model_path = result_path + "yolov7_first_half.pt"

second_half_yaml_path = result_path + "yolov7_second_half.yaml"
second_half_state_dict_path = result_path + "yolov7_second_half_state_dict.pt"
second_half_full_model_path = result_path + "yolov7_second_half.pt"


cut_at_layer = 24
#cut_at_layer = 11

# function for creating yaml files for both halfs of the model
def create_model_halfs_yaml(_yaml_file, _cut_at_layer):
    print("-------------------")
    print("create yaml files for both halfs of the model", "\noriginal yaml file:")
    print(_yaml_file)

    _first_half_yaml = copy(_yaml_file)
    _sec_half_yaml = copy(_yaml_file)

    print("\n-------------------")
          
    len_backbone = len(_first_half_yaml['backbone'])
    if len_backbone > _cut_at_layer:
        print("remove head and only keep", _cut_at_layer, "backbone layers")
        #del _first_half_yaml['head']
        _first_half_yaml['head'] = []
        _first_half_yaml['backbone'] = _first_half_yaml['backbone'][:_cut_at_layer]
    else:
        print("error not implemented yet!!!")
        quit()

    print("first half yaml file:\n", _first_half_yaml)
    print("\n-------------------")

    _sec_half_yaml['backbone'] = _sec_half_yaml['backbone'][_cut_at_layer:]
    _sec_half_yaml['ch'] = 512  #_first_half_yaml['backbone'][-1][-1][0]


    print("-------------------------------------------------------------")
    print("WARNING! CH is a fix number.. find correct number!!")
    print("FIND FIX FOR CHANNEL NUMBER!!!!!!!!!!!!!")
    print("-------------------------------------------------------------")

    for i, item in enumerate(_sec_half_yaml['backbone']):
        if type(item[0]) == type(1) and item[0] > 0 :
            _sec_half_yaml['backbone'][i][0] -= _cut_at_layer
        elif type(item[0]) == type([1,1]):
            new_list = []
            for sub_item in item[0]:
                if sub_item > 0:
                    new_list.append(sub_item - _cut_at_layer)
                else:
                    new_list.append(sub_item)
            item[0]  = new_list

    for i, item in enumerate(_sec_half_yaml['head']):
        if type(item[0]) == type(1) and item[0] > 0 :
            _sec_half_yaml['head'][i][0] -= _cut_at_layer
        elif type(item[0]) == type([1,1]):
            new_list = []
            for sub_item in item[0]:
                if sub_item > 0:
                    new_list.append(sub_item - _cut_at_layer)
                else:
                    new_list.append(sub_item)
            item[0]  = new_list


    print("second half yaml file:", _sec_half_yaml)
    print("saving both yaml to split_model folder")

    with open(first_half_yaml_path, 'w') as output_file:
        yaml.dump(_first_half_yaml, output_file, default_flow_style=True)

    with open(second_half_yaml_path, 'w') as output_file:
        yaml.dump(_sec_half_yaml, output_file, default_flow_style=True)


# function for creating state dicts for both halfs of the model
def create_model_halfs_state_dicts(yolov7_model, _cut_at_layer):
    print("\n-------------------")
    print("create state dicts for both halfs of the model")
    print("the state dict of the original model has a length of: ", len(yolov7_model.state_dict()))
   
    _first_half_model = deepcopy(yolov7_model)
    _sec_half_model = deepcopy(yolov7_model)

    del(list(_first_half_model.children())[0][_cut_at_layer:])
    del(list(_sec_half_model.children())[0][:_cut_at_layer])

    print("the state dict of the first half has a length of: ", len(_first_half_model.state_dict()))
    print("the state dict of the second half has a length of: ", len(_sec_half_model.state_dict()))

    torch.save(_first_half_model.state_dict(), first_half_state_dict_path)
    torch.save(_sec_half_model.state_dict(), second_half_state_dict_path)

    del _first_half_model, _sec_half_model


# function for preparing the test image
def prepare_test_image(im0):
    print("\n-------------------")
    print("load test image to compare with original yolov7 model")
    
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

    # if weights are not present, download them

    
    # load the original yolov7 model
    ckpt = torch.load(weights, weights_only=False, map_location = device)
    yolov7_model = ckpt['model'].float().eval()  # FP32 model

    # create yaml files and state dicts for both halfs of the model
    create_model_halfs_yaml(yolov7_model.yaml, cut_at_layer)
    create_model_halfs_state_dicts(yolov7_model, cut_at_layer)

    # to verify the split model, load the yaml files, apply the state dicts and infere on the test image
    firsModel = Model(first_half_yaml_path)
    secModel = Model(second_half_yaml_path)

    firsModel.load_state_dict(torch.load(first_half_state_dict_path, weights_only=True))
    firsModel.float().eval()

    secModel.load_state_dict(torch.load(second_half_state_dict_path, weights_only=True))
    secModel.float().eval()

    m = secModel.model[-1]  # Detect() module
    m1 = yolov7_model.model[-1]

    s = 256  # 2x min stride
    m.stride = m1.stride
    #check_anchor_order(m)
    m.anchors = m1.anchors
    secModel.stride = m.stride

    for i in range(len(m.m)):
        m.m[i].bias = m1.m[i].bias

    torch.save(firsModel, first_half_full_model_path)
    torch.save(secModel, second_half_full_model_path)

    print("-------------------")
    print("compare original and split model")
    print("")

    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    
    pred_original = yolov7_model(img)[0]
    pred_original_nms = copy(non_max_suppression(pred_original))

    print("Original:" ,len(pred_original_nms[-1]), "detection/s in this image:")
    for i, det in enumerate(pred_original_nms):

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            print("Class:", int(cls)," conf:", float(conf), "xywh:", xywh)
            label = f'{int(cls)} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=(255,0,0), line_thickness=1)


    mid_pred = firsModel(img)
    pred = secModel(mid_pred)[0]
    pred_nms = copy(non_max_suppression(pred))

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
    cv2.imwrite(result_path + "test.png", img0)

    # create and save the graph of the original and both parts of the split model 
    draw_graph(yolov7_model, input_data=img, expand_nested=True, save_graph=True, filename= result_path + "yolov7_graph")
    draw_graph(firsModel, input_data=img, expand_nested=True, save_graph=True, filename= result_path + "yolov7_first_half_graph")
    draw_graph(secModel, input_data=mid_pred, expand_nested=True, save_graph=True, filename= result_path + "yolov7_second_half_graph")






