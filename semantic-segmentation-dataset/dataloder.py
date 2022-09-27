from email.mime import image
import os
import torch
import torchvision
from d2l import torch as d2l

d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
#下载数据集
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
#读取所有voc图像并标注
def read_voc_images(voc_dir,is_train = True):
    txt_fname = os.path.join(voc_dir,'ImageSets','Segmentation','train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname,'r') as f:
        images = f.read().split()
    features,labels =[],[]
    for i,fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features,labels

train_features, train_labels = read_voc_images(voc_dir, True)

#列举RGB颜色值和类名
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

#定义从RGB值到VOC类别索引的映射
def voc_colormap2label():
    colormap2label = torch.zeros(256**3,dtype=torch.long)