import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize(img, label):
    img = img.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    print(np.unique(label))
    plt.figure(figsize= (15,15))
    plt.subplot(1,5,1)
    plt.imshow(img[...,0], cmap='gray')
    plt.subplot(1,5,2)
    plt.imshow(img[...,1], cmap='gray')
    plt.subplot(1,5,3)
    plt.imshow(img[...,2], cmap='gray')
    plt.subplot(1,5,4)
    plt.imshow(img[...,3], cmap='gray')
    plt.subplot(1,5,5)
    plt.imshow(label, cmap='gray')
    
    
def log(mode, epoch, i, loss, l0, l1, l2, l3):
    loss, l0, l1, l2, l3 = [round(j/i,3) for j in [loss, l0, l1, l2, l3]]
    ET = l3
    TC = round((l1 + l2)/2,3)
    WT = round((l1 + l2 + l3)/3,3)
    print(f"{mode} | Epoch: {epoch+1} | Iter: {i+1} Overall Loss: {loss} | BackGround: {l0} | ET : {ET} | TC : {TC} | WT : {WT}")