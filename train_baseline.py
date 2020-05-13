import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from _utils import *
from _dataset import Image_Dataset 
from _model import AE

same_seeds(0)

train_x_name = sys.argv[1]
checkpoint_name = sys.argv[2]
prediction_name = sys.argv[3]

trainX = np.load(train_x_name)
valX = np.load('valX.npy')
valY = np.load('valY.npy')
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

model.train()
n_epoch = 100

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

# 主要的訓練過程
best_acc = 0
for epoch in range(n_epoch):
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    err = 0
    n = 0
    for x in img_dataloader:
        x = x.cuda()
        _, rec = model(x)
        err += torch.nn.MSELoss(reduction='sum')(x, rec).item()
        n += x.flatten().size(0)
    latents = inference(X=valX, model=model)
    pred, X_embedded = predict(latents)
    acc = cal_acc(valY, pred)
    if acc > best_acc:
        best_acc = acc
        print('upd {} -> {}'.format(best_acc, acc))
        torch.save(model.state_dict(), checkpoint_name)
            
    print('epoch [{}/{}], acc{:.5f}, loss:{:.5f}'.format(epoch+1, n_epoch, acc, loss.data))

model.load_state_dict(torch.load(checkpoint_name))
model.eval()

# 預測答案
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)

# 將預測結果存檔，上傳 kaggle
#save_prediction(pred, 'prediction.csv')

# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
save_prediction(invert(pred), prediction_name)
