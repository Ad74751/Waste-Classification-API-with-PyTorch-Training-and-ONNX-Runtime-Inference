DATA_DIR = './dataset/raw'
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'model.pth'
ONNX_SAVE_PATH = 'model.onnx'
IMG_SIZE = 128
NUM_CLASSES = 9
EARLY_STOPPING_PATIENCE = 3

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
