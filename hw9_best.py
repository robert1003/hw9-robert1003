import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from _utils import *
from _dataset import *
from _model import *

train_x_name = sys.argv[1]
checkpoint_name = sys.argv[2]
prediction_name = sys.argv[3]

trainX = np.load(train_x_name)
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

model = AE_best().cuda()
same_seeds(1003)

model.load_state_dict(torch.load(checkpoint_name))
model.eval()

latents = inference(X=trainX, model=model)
pred, X_embedded = predict_best(latents)

save_prediction(pred, prediction_name)
