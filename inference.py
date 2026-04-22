import torch

from networks.GSformer import GSformer,GDGSformer
from networks.models_config import parse_option

import cv2

from PIL import Image
import torchvision.transforms as transforms

args,config = parse_option("test")

#set device for test

model = GDGSformer(config)

print('NOW USING RGB BackBone:',args.backbone)

if args.test_model:
    state_dict = torch.load(args.test_model+'/ckpt/Best_mae_test.pth',map_location=torch.device('cpu'))

    model_dict = {}

    for k,v in state_dict.items():
        if k.startswith('module'):
            model_dict[k[7:]] = v
        else:
            model_dict[k] = v
    
    msg = model.load_state_dict(model_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.test_model+'/ckpt/Best_mae_test.pth', msg))

else:
    print("test_model=path/to/model.pth")
    #exit(0)

model.eval()


rgb_transform = transforms.Compose([
    transforms.Resize(( config.DATA.IMG_SIZE,  config.DATA.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
binary_transform = transforms.Compose([
    transforms.Resize(( config.DATA.IMG_SIZE,  config.DATA.IMG_SIZE)),
    transforms.ToTensor()])

gt_resize_transform = transforms.Compose([
    transforms.Resize(( config.DATA.IMG_SIZE,  config.DATA.IMG_SIZE),interpolation=transforms.InterpolationMode.NEAREST)
    ])

resize_transform = transforms.Compose([
    transforms.Resize(( config.DATA.IMG_SIZE,  config.DATA.IMG_SIZE))
    ])

def rgb_loader(path):
    return Image.open(path).convert('RGB')

def binary_loader(path):
    return Image.open(path).convert('L')

image = rgb_loader('./dataset/RGBD_dataset/train/COME/RGB/COME_Train_4.jpg')
gray = binary_loader('./dataset/RGBD_dataset/train/COME/RGB/COME_Train_4.jpg')
depth = binary_loader('./dataset/RGBD_dataset/train/COME/depth/COME_Train_4.png')
gt = rgb_loader('./dataset/RGBD_dataset/train/COME/GT/COME_Train_4.png')
namlab = rgb_loader('./dataset/RGBD_dataset/train/COME/namlab40/COME_Train_4.png')
bound = rgb_loader('./dataset/RGBD_dataset/train/COME/bound/COME_Train_4.png')

resize_transform(image).save("./fork/RGB.png")
resize_transform(depth).save("./fork/depth.png")
gt_resize_transform(gt).save("./fork/GT.png")
resize_transform(namlab).save("./fork/namlab.png")
resize_transform(bound).save('./fork/bound.png')

image = rgb_transform(image)
depth = binary_transform(depth)
gray = binary_transform(gray)

def sig2map(sig,fname):
    #res = F.interpolate(sig, size=(360,640), mode='bilinear', align_corners=False)
    res = sig.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)*255
    cv2.imwrite(fname,res)


with torch.no_grad():

    image = image.unsqueeze(0)
    depth = depth.unsqueeze(0)
    gray = gray.unsqueeze(0)

    s1,s2,s3,s4,edge_sod,edge_rgb,edge_depth = model(image, torch.cat((depth,gray),dim=1))
    
    
    print(s1.shape,s2.shape,s3.shape,s4.shape)
    # sig2map(s1,"./fork/s1.png")
    # sig2map(s2,"./fork/s2.png")
    # sig2map(s3,"./fork/s3.png")
    # sig2map(s4,"./fork/s4.png")
    # sig2map(edge_sod,"./fork/edge_sod.png")
    # sig2map(edge_rgb,"./fork/edge_rgb.png")
    # sig2map(edge_depth,"./fork/edge_depth.png")


print('Test Done!')


