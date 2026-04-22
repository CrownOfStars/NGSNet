import os
from PIL import Image,ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import cv2
import torch.nn.functional as F
import math
"""
 TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 distributed.py --backbone segswin-base --texture /namlab40/ --mfusion HAIM --lr 3e-4 --train_batch 48
"""
# several data augumentation strategies
def random_flip(*images):

    if random.randint(0, 1):
        for image in images:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if random.randint(0,1):
        for image in images:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)    

    return images


def random_crop(*images):
    
    image_width,image_height = images[0].size[0], images[0].size[1]
    border_width,border_height = image_width*0.1,image_height*0.1
    crop_win_width = np.random.randint(image_width - border_width, image_width)
    crop_win_height = np.random.randint(image_height - border_height, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for image in images:
        image = image.crop(random_region)

    return images



def random_rotation(*images):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for image in images:
            image = image.rotate(random_angle, mode)

    return images


def color_enhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def image_suffix(f):
    return f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')


class SalObjTrainDataset(data.Dataset):
    def __init__(self, dataset_root, texture_type,trainsize):

        image_root = dataset_root + '/RGB/'
        depth_root = dataset_root + '/depth/'
        bound_root = dataset_root + '/bound/'
        if texture_type:
            texture_root = dataset_root + texture_type
        else:
            texture_root = bound_root
        gt_root = dataset_root + '/GT/'

        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
        self.texs = sorted([texture_root+f for f in os.listdir(texture_root) if image_suffix(f)])
        self.bounds = sorted([bound_root+f for f in os.listdir(bound_root) if image_suffix(f)])
        
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} train data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.logistic_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.rgb_loader(self.depths[index])
        
        gt = self.binary_loader(self.gts[index])
        texture = self.binary_loader(self.texs[index])
        bound = self.binary_loader(self.bounds[index])

        image, gt, depth, texture, bound = random_flip(image, gt, depth, texture, bound)
        image, gt, depth, texture, bound = random_crop(image, gt, depth, texture, bound)
        image, gt, depth, texture, bound = random_rotation(image, gt, depth, texture, bound)
        image = color_enhance(image)
        
        
        image = self.rgb_transform(image)
        depth = self.logistic_transform(depth)
        gt = self.binary_transform(gt)
        bound = self.binary_transform(bound)
        texture = self.binary_transform(texture)
        
        texture = F.avg_pool2d(F.max_pool2d(texture,kernel_size=3,stride=1,padding=1),kernel_size=3,stride=1,padding=1)
        return image, gt, depth, texture,bound


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')


    def __len__(self):
        return self.size

class SalObjValDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize):

        image_root = dataset_root + '/RGB/'
        depth_root = dataset_root + '/depth/'
        gt_root = dataset_root + '/GT/'
        
        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
       
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} val data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.rgb_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])

        image = self.rgb_transform(image)
        depth = self.depth_transform(depth)
        gt = self.binary_transform(gt)

        return image, gt, depth

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size

# test dataset and loader
class SalObjTestDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize):

        image_root = dataset_root + '/RGB/'
        depth_root = dataset_root + '/depth/'
        gt_root = dataset_root + '/GT/'

        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
        
        
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} test data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
           
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = cv2.imread(self.gts[index],cv2.IMREAD_GRAYSCALE)
        sz = gt.shape
        depth = self.rgb_loader(self.depths[index])
        h,w = sz

        resize = transforms.Resize([384,384])

        image = resize(self.rgb_transform(image))
        depth = resize(self.binary_transform(depth))

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return  image, gt, depth, sz, name


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size


class GDSalObjTrainDataset(data.Dataset):
    def __init__(self, dataset_root, texture_type,trainsize):

        image_root = dataset_root + '/RGB/'
        depth_root = dataset_root + '/depth/'
        bound_root = dataset_root + '/bound/'
        if texture_type:
            texture_root = dataset_root + texture_type
        else:
            texture_root = bound_root
        gt_root = dataset_root + '/GT/'

        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
        self.texs = sorted([texture_root+f for f in os.listdir(texture_root) if image_suffix(f)])
        self.bounds = sorted([bound_root+f for f in os.listdir(bound_root) if image_suffix(f)])
        
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} train data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.logistic_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gray = self.binary_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        texture = self.binary_loader(self.texs[index])
        bound = self.binary_loader(self.bounds[index])

        image, gt, depth, gray, texture, bound = random_flip(image, gt, depth, gray, texture, bound)
        image, gt, depth, gray, texture, bound = random_crop(image, gt, depth, gray, texture, bound)
        image, gt, depth, gray, texture, bound = random_rotation(image, gt, depth, gray, texture, bound)
        image = color_enhance(image)
        
        
        image = self.rgb_transform(image)
        depth = self.logistic_transform(depth)
        gray = self.logistic_transform(gray)
        gt = self.binary_transform(gt)
        bound = self.binary_transform(bound)
        texture = self.binary_transform(texture)
        
        texture = F.avg_pool2d(F.max_pool2d(texture,kernel_size=3,stride=1,padding=1),kernel_size=3,stride=1,padding=1)
        return image, gt, depth, gray, texture,bound


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')


    def __len__(self):
        return self.size

class GDSalObjValDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize):

        image_root = dataset_root + '/RGB/'
        depth_root = dataset_root + '/depth/'
        gt_root = dataset_root + '/GT/'
        
        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
       
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} val data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        gray = self.binary_loader(self.images[index])

        image = self.rgb_transform(image)
        depth = self.depth_transform(depth)
        gray = self.depth_transform(gray)
        gt = self.binary_transform(gt)

        return image, gt, depth, gray

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size

# test dataset and loader
class GDSalObjTestDataset(data.Dataset):
    def __init__(self, dataset_root, trainsize):

        image_root = dataset_root + '/RGB/'
        depth_root = dataset_root + '/depth/'
        gt_root = dataset_root + '/GT/'

        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
        
        
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} test data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
           
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = cv2.imread(self.gts[index],cv2.IMREAD_GRAYSCALE)
        sz = gt.shape
        gray = self.binary_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        h,w = sz

        resize = transforms.Resize([384,384])

        image = resize(self.rgb_transform(image))
        depth = resize(self.binary_transform(depth))
        gray = resize(self.binary_transform(gray))

        
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return  image, gt, gray, depth, sz, name


    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size


def get_loader(dataset_root, batchsize, trainsize,dist = False, texture_type = None, ds_type='train'):
    if ds_type == 'train':   
        train_dataset = data.ConcatDataset([SalObjTrainDataset(dataset_root+'/'+dataset,texture_type,trainsize) for dataset in os.listdir(dataset_root)])
        data_loader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batchsize,
                                      
                                      num_workers=4,sampler= data.distributed.DistributedSampler(train_dataset) if dist else None)
        
        return data_loader
    
    elif ds_type == 'val':   
        train_dataset = data.ConcatDataset([SalObjValDataset(dataset_root+'/'+dataset,trainsize) for dataset in os.listdir(dataset_root)])
        data_loader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batchsize,
                                      num_workers=4,sampler= data.distributed.DistributedSampler(train_dataset) if dist else None)
        return data_loader
    elif ds_type == 'test':

        dataset = SalObjTestDataset(dataset_root,trainsize) 
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=1,
                                      num_workers=4)
        
        return data_loader
    else:
        raise NotImplementedError("no such dataset")

def get_gd_loader(dataset_root, batchsize, trainsize,dist = False, texture_type = None, ds_type='train'):
    if ds_type == 'train':   
        train_dataset = data.ConcatDataset([GDSalObjTrainDataset(dataset_root+'/'+dataset,texture_type,trainsize) for dataset in os.listdir(dataset_root)])
        data_loader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batchsize,
                                      
                                      num_workers=4,sampler= data.distributed.DistributedSampler(train_dataset) if dist else None)
        
        return data_loader
    
    elif ds_type == 'val':   
        train_dataset = data.ConcatDataset([GDSalObjValDataset(dataset_root+'/'+dataset,trainsize) for dataset in os.listdir(dataset_root)])
        data_loader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batchsize,
                                      num_workers=4,sampler= data.distributed.DistributedSampler(train_dataset) if dist else None)
        return data_loader
    elif ds_type == 'test':

        dataset = GDSalObjTestDataset(dataset_root,trainsize) 
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=1,
                                      num_workers=4)
        
        return data_loader
    else:
        raise NotImplementedError("no such dataset")

if __name__ == "__main__":
    train_loader = get_loader('./dataset/RGBD_dataset/train/', 4, 224,False,'/namlab30/',"train")
    for i, (images, gts,depths,texs,bounds) in enumerate(train_loader, start=1):
        print(images.shape,depths.shape,gts.shape,texs.shape,bounds.shape)
    