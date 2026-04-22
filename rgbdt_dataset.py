import os
from PIL import Image,ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import cv2


# several data augumentation strategies
def random_flip(image, label, depth, bound, texture):

    if random.randint(0, 1):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        bound = bound.transpose(Image.FLIP_LEFT_RIGHT)
        texture = texture.transpose(Image.FLIP_LEFT_RIGHT)
    if random.randint(0,1):
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        bound = bound.transpose(Image.FLIP_TOP_BOTTOM)
        texture = texture.transpose(Image.FLIP_TOP_BOTTOM)
    return image, label, depth, bound, texture


def random_crop(image, label, depth, bound, texture):
    border = 30
    image_width,image_height = image.size[0], image.size[1]
    
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region),bound.crop(random_region),texture.crop(random_region)


def random_rotation(image, label, depth,bound,texture):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        bound = bound.rotate(random_angle, mode)
        texture = texture.rotate(random_angle,mode)
    return image, label, depth,bound,texture


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

# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjTrainDataset(data.Dataset):
    def __init__(self, dataset_root, texture_type,trainsize):

        image_root = dataset_root + '/RGB/'
        depth_root = dataset_root + '/depth/'

        if texture_type:
            texture_root = dataset_root + texture_type
        else:
            texture_root = dataset_root + '/namlab60/'
        gt_root = dataset_root + '/GT/'
        bound_root = dataset_root + '/bound/'

        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if image_suffix(f)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if image_suffix(f)])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if image_suffix(f)])
        self.bound = sorted([bound_root + f for f in os.listdir(bound_root) if image_suffix(f)])
        self.texture = sorted([texture_root+f for f in os.listdir(texture_root) if image_suffix(f)])
        
        
        assert len(self.images) == len(self.depths) and len(self.gts) == len(self.images)
        self.size = len(self.images)
        print(f'load {self.size} train data from {dataset_root}')
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])
        bound = self.binary_loader(self.bound[index])
        texture = self.binary_loader(self.texture[index])

        image, gt, depth,bound,texture = random_flip(image, gt, depth,bound,texture)
        image, gt, depth,bound,texture = random_crop(image, gt, depth,bound,texture)
        image, gt, depth,bound,texture = random_rotation(image, gt, depth,bound,texture)
        image = color_enhance(image)
        
        
        image = self.rgb_transform(image)
        gt = self.binary_transform(gt)
        bound = self.binary_transform(bound)
        depth = self.binary_transform(depth)
        texture = self.binary_transform(texture)
        return image, gt, depth,texture,bound


    def rgb_loader(self, path):
        return Image.fromarray(cv2.imread(path))

    def binary_loader(self, path):
        return Image.fromarray(cv2.imread(path,cv2.IMREAD_GRAYSCALE))

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

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
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])

        
        image = self.rgb_transform(image)
        gt = self.binary_transform(gt)
        depth = self.binary_transform(depth)
        return image, gt, depth


    def rgb_loader(self, path):
        return Image.fromarray(cv2.imread(path))

    def binary_loader(self, path):
        return Image.fromarray(cv2.imread(path,cv2.IMREAD_GRAYSCALE))

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
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.binary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = cv2.imread(self.gts[index],cv2.IMREAD_GRAYSCALE)
        sz = gt.shape
        depth = self.rgb_loader(self.depths[index])

        image = self.rgb_transform(image)
        depth = self.binary_transform(depth)

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return  image, gt, depth, sz, name


    def rgb_loader(self, path):
        return Image.fromarray(cv2.imread(path))

    def binary_loader(self, path):
        return Image.fromarray(cv2.imread(path,cv2.IMREAD_GRAYSCALE))

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),Image.NEAREST)

    def __len__(self):
        return self.size

    

# dataloader for training
def get_loader(dataset_root, batchsize, trainsize,dist = False, texture_type = None, ds_type='train'):
    """
        
    """
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
                                      num_workers=4)
        return data_loader
    elif ds_type == 'test':

        dataset = SalObjTestDataset(dataset_root,trainsize) 
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=1,
                                      num_workers=4)
        
        return data_loader
    else:
        raise NotImplementedError("no such dataset")




def get_demo_input(rgb_path,gt_path,depth_path,testsize):
    rgb_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    binary_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor()])
      
    def rgb_loader(path):
        return Image.open(path).convert('RGB')

    def binary_loader(path):
        return Image.fromarray(cv2.imread(path,cv2.IMREAD_GRAYSCALE))

    image = rgb_loader(rgb_path)
    gt = binary_loader(gt_path)
    depth = rgb_loader(depth_path)

    image = binary_transform(image)
    gt = binary_transform(gt)
    depth = binary_transform(depth)

    return  image, gt,depth

