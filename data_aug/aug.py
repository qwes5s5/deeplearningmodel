
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('C:/Users/wenlei/Desktop/model/deeplearningmodel/cat1.jpg')
d2l.plt.imshow(img)
d2l.plt.show()
#aug图片增广办法，scale放大倍数
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

apply(img, torchvision.transforms.RandomHorizontalFlip())
d2l.plt.show()

apply(img, torchvision.transforms.RandomVerticalFlip())

#(200,200)最后的大小，scale剪裁范围从10%到100%，ratio高宽比1:2或2:1
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

#brightness亮度 contrast对比度 saturation饱和度 hue色调 0.5表示上下浮动50%
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))

augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)



