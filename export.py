import torch
from model import DispNet


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_INPUT_CHANNELS = 3


x = torch.randn(1, IMAGE_INPUT_CHANNELS * 2, IMAGE_HEIGHT, IMAGE_WIDTH)
model = DispNet()
torch.onnx.export(DispNet(), x, 'DispNet.onnx')
