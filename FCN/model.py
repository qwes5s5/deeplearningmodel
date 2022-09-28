from ast import Num
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
#调用resnet除了最后两层之外的层
pretrained_net = torchvision.models.resnet18(pretrained=True)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
#构建FCN网络
num_classes = 21
net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))
net.add_module('transpose_conv',nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))
#初始化转置卷积层
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

W = bilinear_kernel(num_classes,num_classes,64)
net.transpose_conv.weight.data.copy_(W)
#读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
#训练
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)#输出的是矩阵，需要对高和宽求均值

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
#预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
#将类别映射回颜色
def label2img(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP,device=devices[0])
    X = pred.long()
    return colormap[X,:]
#开始预测
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images,test_labels = d2l.read_voc_images(voc_dir,False)
n,imgs=4,[]
for i in range(n):
    crop_rect = (0,0,320,480)
    X = torchvision.funtional.crop(test_images[i],*crop_rect)
    pred = label2img(predict(X))
    imgs += [X.permute(1,2,0),pred.cpu(),torchvision.funtional.crop(test_labels[i],*crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);