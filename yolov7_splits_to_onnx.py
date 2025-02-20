import torch
import torch.nn as nn
import onnx
import sys

sys.path.append('yolov7')

import models
from utils.activations import Hardswish, SiLU
from utils.torch_utils import select_device

result_path = "split_yolov7_model/"

first_half_path = result_path + "yolov7_first_half.pt"
second_half_path = result_path + "yolov7_second_half.pt"

def toOnnx(outname, model, input, input_names, output_names):
	# Update model
	for k, m in model.named_modules():
		m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
		if isinstance(m, models.common.Conv):  # assign export-friendly activations
			if isinstance(m.act, nn.Hardswish):
				m.act = Hardswish()
			elif isinstance(m.act, nn.SiLU):
				m.act = SiLU()

	# ONNX export
	try:
		print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
		f = outname # filename
		model.eval()
		dynamic_axes = None

		torch.onnx.export(model, input, f, verbose=False, opset_version=12, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

		# Checks
		onnx_model = onnx.load(f)  # load onnx model
		onnx.checker.check_model(onnx_model)  # check onnx model

		onnx.save(onnx_model,f)
		print('ONNX export success, saved as %s' % f)

	except Exception as e:
		print('ONNX export failure: %s' % e)


# main function
if __name__ == '__main__':
	device = select_device("cpu")

	firsModel = torch.load(first_half_path, weights_only=False, map_location = device).float().eval()
	secModel = torch.load(second_half_path, weights_only=False, map_location = device).float().eval()

	secModel.model[-1].export = True

	batch_size = 1

	# Input
	inputP1 = torch.zeros(batch_size, 3, *[640, 640]).to(device)
	inputP2 = torch.zeros(batch_size, 512, *[80, 80]).to(device)

	toOnnx("yolov7_first_half.onnx", firsModel, inputP1, ['images'], ['output'])
	toOnnx("yolov7_second_half.onnx", secModel, inputP2, ['p1_data'], ['output'])
