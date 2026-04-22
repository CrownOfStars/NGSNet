import os
import cv2
from tqdm import tqdm
PATH = "/home/data1/ShiqiangShu/NAMSwinNet/dataset/RGBD_dataset/train/NJUNLPR/RGB/"

files = os.listdir(PATH)

def shape2int(shape):
    return shape[0]*1e4+shape[1]

shapeset = set()

for file in tqdm(files):
    shapeset.add(shape2int(cv2.imread(PATH+file).shape))

print(list(shapeset))

"""
# define sampler sample same shape image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def show_image_batch(img_list, title=None):
    num = len(img_list)
    fig = plt.figure()
    for i in range(num):
        ax = fig.add_subplot(1, num, i+1)
        ax.imshow(img_list[i].numpy().transpose([1,2,0]))
        ax.set_title(title[i])

    plt.show()

#  do not do randomCrop to show that the custom collate_fn can handle images of different size
train_transforms = transforms.Compose([transforms.Scale(size = 224),
                                       transforms.ToTensor(),
                                       ])

# change root to valid dir in your system, see ImageFolder documentation for more info
train_dataset = datasets.ImageFolder(root="/hd1/jdhao/toyset",
                                     transform=train_transforms)

trainset = DataLoader(dataset=train_dataset,
                      batch_size=4,
                      shuffle=True,
                      collate_fn=my_collate, # use custom collate function here
                      pin_memory=True)

trainiter = iter(trainset)
imgs, labels = trainiter.next()

# print(type(imgs), type(labels))
show_image_batch(imgs, title=[train_dataset.classes[x] for x in labels])
"""

#TODO: similar shape sampler:
