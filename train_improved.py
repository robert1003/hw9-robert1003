import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
from torch import optim
from torch.utils.data import DataLoader
from _utils import *
from _dataset import *
from _model import *

same_seeds(1003)

train_x_name = sys.argv[1]
checkpoint_name = sys.argv[2]
prediction_name = sys.argv[3]

trainX = np.load(train_x_name)
valX = np.load('valX.npy')
valY = np.load('valY.npy')
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

model = AE_best().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())# lr=5e-5, weight_decay=1e-5)

model.train()
n_epoch = 200

img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s', 
    handlers=[logging.FileHandler(checkpoint_name[:-4] + '.log', 'w'), logging.StreamHandler(sys.stdout)]
)

best_acc = 0
for epoch in range(n_epoch):
    model.train()
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    latents = inference(X=valX, model=model)
    pred, X_embedded = predict_best(latents)
    acc = cal_acc(valY, pred)
    if acc > best_acc:
        logging.info('upd {} -> {}'.format(best_acc, acc))
        best_acc = acc
        torch.save(model.state_dict(), checkpoint_name)
        
    logging.info('best: {:.5f}'.format(best_acc))
    logging.info('epoch [{}/{}], acc: {:.5f}, loss:{:.5f}'.format(epoch+1, n_epoch, acc, loss.data))

logging.info('final: {:.5f}'.format(best_acc))

model.load_state_dict(torch.load(checkpoint_name))
model.eval()

latents = inference(X=trainX, model=model)
pred, X_embedded = predict_best(latents)

save_prediction(invert(pred), prediction_name)
