import cv2, sys, os
import onnxruntime as ort
from PIL import Image
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch
from copy import deepcopy, copy

device ='cpu'

yoloGitPath = "yolov7/"
sys.path.append('yolov7')

image_path = yoloGitPath + "inference/images/horses.jpg"

from utils.general import non_max_suppression, xyxy2xywh, scale_coords
from utils.plots import plot_one_box

result_path = "split_yolov7_model/"
first_half_path = result_path + "yolov7_first_half.pt"
second_half_path = result_path + "yolov7_second_half.pt"

height = 640
width = 640

def normalize(image, normalization_params):
    mean, std = normalization_params
    return (image - list(mean)) / list(std)

def letterbox(img, height=608, width=1088, centered=True,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (int(shape[1] * ratio), int(shape[0] * ratio))  # new_shape = [width, height]
    new_width = new_shape[0]
    dw = (width - new_width) / 2 if centered else (width - new_width)  # width padding
    new_height = new_shape[1]
    dh = (height - new_height) / 2 if centered else (height - new_height)  # height padding
    if centered:
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
    else:
        top, bottom = 0, dh
        left, right = 0, dw
    img = cv2.resize(img, new_shape,
                     interpolation=(cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR))  # resized, no border
    # cv2 uses bgr format, need to switch the color
    color_bgr = color[::-1]
    img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                             value=color_bgr)  # padded rectangular
    return img, new_width, new_height


# Load the model and create InferenceSession
model_path = "yolov7_first_half.onnx"
print("loading", model_path, "to create the input tensor")
session = ort.InferenceSession(model_path)

img0 = cv2.imread(image_path)  # BGR

img = np.array(Image.open(image_path))
image_height = img.shape[0]
image_width = img.shape[1]


image, new_width, new_height = letterbox(img, height, width, color=[114] * 3, centered=True)

image = np.ascontiguousarray(image)

normalization_params = [[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]]
image = normalize(image, normalization_params)
image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
image = np.expand_dims(image, axis=0)

outputs = session.run(None, {"images": image.astype(np.float32)})

firsModel = torch.load(first_half_path, weights_only=False, map_location= device).float().eval()

t = torch.from_numpy(copy(image))
t= t.float()

with torch.no_grad():
    mid_pred = firsModel(t)[0]

t2 = torch.from_numpy(copy(outputs[0]))
t2 = t2.float()

secModel = torch.load(second_half_path, weights_only=False, map_location = device).float().eval()

with torch.no_grad():
    pred = secModel(t2)[0]

pred_nms = non_max_suppression(pred)
    
gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

print("Split model:",len(pred_nms[-1]), "detection/s in this image:")
for i, det in enumerate(pred_nms):

    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords([640,640], det[:, :4], img0.shape).round()

    for *xyxy, conf, cls in reversed(det):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        print("Class:", int(cls)," conf:", float(conf), "xywh:", xywh)
        label = f'{int(cls)} {conf:.2f}'
        plot_one_box(xyxy, img0, label=label, color=(0,0,255), line_thickness=1)


    # save the test image with all boxes drawn 
    cv2.imwrite(result_path + "test_4.png", img0)


print(np.allclose(t2, mid_pred, rtol=1e-05, atol=1e-04))