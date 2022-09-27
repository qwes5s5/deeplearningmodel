import os
from random import shuffle
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
    for i,colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256+colormap[1])*256+colormap[2]]=i
    return colormap2label
#将VOC标签中的RGB值映射到类别0\1\2....
def voc_label_indices(colormap,color2label):
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    idx = ((colormap[:,:,0]*256+colormap[:,:,1])*256+colormap[:,:,2])
    return color2label(idx)

#数据预处理
#随机裁剪特征图和相应标签图像
def voc_rand_crop(feature,label,height,width):
    rect = torchvision.transforms.RandomCrop.get_params(
        feature,(height,width)
    )
    feature = torchvision.transforms.functional.crop(feature,*rect)
    label = torchvision.transforms.functional.crop(label,*rect)
    return feature,label
#自定义语义分割数据集类
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self,is_train,crop_size,voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features,labels = read_voc_images(voc_dir,is_train=is_train)
        self.features = [self.normalize_image(feature)
        for feature in features]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
    def normalize_image(self, img):
        return self.transform(img.float() / 255)
    def filter(self,imgs):
        return[img for img in imgs if(
            img.shape[1]>=self.crop_size[0] and img.shape[2]>=self.crop_size[1]
        )]
    def __getitem__(self,idx):
        feature,label = voc_rand_crop(self.features[idx],self.labels[idx],*self.crop_size)
        return (feature,voc_label_indices(label,self.colormap2label))
    def __len__(self):
        return len(self.features)

#读取数据集
crop_size = (320,480)
voc_train = VOCSegDataset(True,crop_size,voc_dir)
voc_train = VOCSegDataset(False,crop_size,voc_dir)
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
#整合所有组件
def load_data_voc(batch_size,crop_size):
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True,crop_size,voc_dir),batch_size,
        shuffle=True,drop_last=True,num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False,crop_size,voc_dir),batch_size,
        shuffle=True,drop_last=True,num_workers=num_workers
    )
    return train_iter,test_iter
