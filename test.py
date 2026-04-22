import torch
import os
from networks.GSformer import GDGSformer,GSformer
from networks.models_config import parse_option
import SalObjDataset
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import math
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm
import pandas as pd
import numpy as np
#from torch.cuda.amp import autocast, GradScaler

args,config = parse_option("test")

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print('USE GPU:', args.gpu_id)


model = GSformer(config)

print('NOW USING BackBone:',args.backbone)

if not args.test_model:
    print("test_model=path/to/model.pth")
    exit(0)

state_dict = torch.load(args.test_model+'/ckpt/Best_mae_test.pth',map_location=torch.device('cuda'))

model_dict = {}

for k,v in state_dict.items():
    if k.startswith('module'):
        model_dict[k[7:]] = v
    else:
        model_dict[k] = v

model.load_state_dict(model_dict,strict=False)#load the model


save_path = args.test_model+'/save/'

if not os.path.exists(save_path):
    os.mkdir(save_path)


model.cuda()
model.eval()
#scaler = GradScaler()

eval_result = {"MAE":[],
               "maxF":[],
                "avgF":[],
                "wfm": [],
                "sm": [],
                "em": []}


datasets = ["DUT", "DES",  "LFSD",  "NJU2K",  "NLPR",  "SIP",  "SSD",  "STERE", "COME-E", "COME-H", "ReDWeb-S"]



for dataset in datasets:
    
    dataset_root = args.test_path + dataset 

    dataset_save_path = save_path + dataset+'/'
    if not os.path.exists(dataset_save_path):
        os.mkdir(dataset_save_path)

    test_loader = SalObjDataset.get_loader(dataset_root, args.test_batch, config.DATA.IMG_SIZE,ds_type='test')

    mae,fm,sm,em,wfm= cal_mae(),cal_fm(len(test_loader.dataset)),cal_sm(),cal_em(),cal_wfm()
    with torch.no_grad():

        for image, gt,depth,sz,name in tqdm(test_loader):

            gt = gt.numpy().squeeze()/255.0

            image = image.cuda()
            depth = depth.cuda()
            #gray = gray.cuda()

            res = model(image, depth)

            fname = dataset_save_path+name[0]
            
            res = F.interpolate(res[0], size=sz, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(fname, res*255)
            

            mae.update(res,gt)
            sm.update(res,np.where(gt>0.5,1.0,0.0))
            fm.update(res, np.where(gt>0.5,1.0,0.0))
            em.update(res,np.where(gt>0.5,1.0,0.0))
            wfm.update(res,np.where(gt>0.5,1.0,0.0))

        MAE = mae.show()

        maxf,meanf,_,_ = fm.show()
        sm = sm.show()
        em = em.show()
        wfm = wfm.show()
        print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,sm,em))
        eval_result['MAE'].append(MAE)
        eval_result['maxF'].append(maxf)
        eval_result['avgF'].append(meanf)
        eval_result['wfm'].append(wfm)
        eval_result['sm'].append(sm)
        eval_result['em'].append(em)


pd.DataFrame(eval_result,index=datasets).to_csv(args.test_model+'/eval_result.csv')
print('Test Done!')


