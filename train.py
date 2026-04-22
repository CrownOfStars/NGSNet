import os
import torch
from datetime import datetime
import SalObjDataset
import logging
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from networks.models_config import parse_option
from networks.GSformer import GSformer
from tqdm import tqdm
from loss import *
from utils import save_project,clip_gradient
import pickle
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import autocast, GradScaler

args,config = parse_option()


# set the device for training
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print('USE GPU:', args.gpu_id)


print(args)
print(config)


scaler = GradScaler()
model = GSformer(config)
model.cuda()


def log_name(backbone):
    if args.tag:
        log_prefix = args.tag+'GSformer-' 
    else:
        log_prefix = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'GSformer-' 
    if len(backbone) == 1:
        return log_prefix + backbone[0]+"+"+backbone[0]
    else:
        return log_prefix + backbone[0]+"+"+backbone[1]

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

log_path = args.log_path + log_name(args.backbone) + '/'
src_path = log_path+'src/'
fig_path = log_path+'fig/'
ckpt_path = log_path +'ckpt/'


if not os.path.exists(log_path):
    os.makedirs(log_path)
    os.makedirs(fig_path)
    os.makedirs(ckpt_path)
    os.makedirs(src_path)


#save src code and config
save_project(src_path,['./','./networks'])

with open(log_path+"args.json", mode="w") as f:
    json.dump(args.__dict__, f, indent=4)

config.dump(stream=open(log_path+"config.yaml", "w"))

logging.basicConfig(filename=log_path+'log.txt', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

logging.info("Config:")
logging.info(config)
logging.info("Args:")
logging.info(args)

step = 0

mae_list = []
miou_list = []


best_mae = 0
best_epoch = 0


def train(train_loader, model, optimizer, scheduler, epoch):
    global step
    model.train()
    total_step = len(train_loader)
    loss_all = 0
    epoch_step = 0

    try:
        
        for i, (images, gts,depths,texs,bounds) in enumerate(train_loader, start=1):

            with autocast():
                optimizer.zero_grad()
                images = images.cuda()
                depths = depths.cuda()
                gts = gts.cuda()
                texs = texs.cuda()
                bounds = bounds.cuda()
                s1, s2, s3, s4, edge1, edge2, edge3 = model(images, depths)

                if args.texture:
                    loss = GTSupervision(s1,s2,s3,s4,gts) + EdgeSupervision(edge1,bounds)+\
                    NAMLABSupervision(edge2,texs) + EdgeSupervision(edge3, bounds)
                else:
                    loss = GTSupervision(s1,s2,s3,s4,gts) + EdgeSupervision(edge1,bounds)+\
                    EdgeSupervision(edge3, bounds)                    

            scaler.scale(loss).backward()  # 将张量乘以比例因子，反向传播
            clip_gradient(optimizer, args.clip) # 裁剪梯度
            scaler.step(optimizer)  # 将优化器的梯度张量除以比例因子。
            scaler.update()  # 更新比例因子

            step += 1
            epoch_step = epoch_step +1
            loss_all = loss_all + loss.item()
            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch+1, args.max_epoch, i, total_step, loss.item()))
                
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch+1, args.max_epoch, i, total_step, loss.item()))
                
        scheduler.step()
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, LR: {}'.format(epoch+1, args.max_epoch, loss_all, optimizer.param_groups[0]['lr']))

        #if (epoch+1) % 10 == 0 or (epoch+1) == args.max_epoch:
        #    torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch+1))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# val function
def val(val_loader, model, epoch):
    global best_mae, best_epoch

    model.eval()
    with torch.no_grad():
        mae_loss = torch.nn.L1Loss()
        sum_IoU = 0.0
        sum_mae = 0.0
        for _, (images, gts,depths) in enumerate(val_loader, start=1):
            
            gts = gts.cuda()
            images = images.cuda()
            depths = depths.cuda()

            res = model(images, depths)
            res = torch.sigmoid(res[0])

            sum_mae += mae_loss(res,gts).item()*len(gts)
            sum_IoU += IoU(res,gts).item()


        
        mae_list.append(sum_mae/len(val_loader.dataset))
        miou_list.append(sum_IoU/len(val_loader.dataset))

        print("MIoU:",miou_list[-1])
        print("MAE:", mae_list[-1])

        if epoch == 0 or mae_list[-1] < best_mae:
            best_mae = mae_list[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path + "Best_mae_test.pth")
            print('update best epoch to epoch {}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae_list[-1], best_epoch,best_mae ))
        print("Best MAE",best_mae)
        print("Best Epoch",best_epoch)
        
        data = {
            "MAE":mae_list,
            "MIoU":miou_list,
        }
        plt.clf()
        plt.plot(mae_list)
        plt.savefig(fig_path+"mae.png")
        plt.clf()
        plt.plot(miou_list)
        plt.savefig(fig_path+"miou.png")

        pickle.dump(data,open(fig_path+'data.pkl','wb'))


def warmup_init(model):
    for param in model.encoderR.parameters():
        param.requires_grad = False
    train_loader = SalObjDataset.get_loader(args.train_root, args.train_batch, config.DATA.IMG_SIZE,False,args.texture,"train")
    val_loader = SalObjDataset.get_loader(args.val_root, args.train_batch, config.DATA.IMG_SIZE,False,args.texture,ds_type="val")

    optimizer = optim.Adam(params = model.parameters(), lr = args.lr, betas=[0.9,0.999], eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.decay_epoch//4, gamma=args.gamma)
    return train_loader,val_loader,optimizer,scheduler

def finetune_init(model):
    for param in model.encoderR.parameters():
        param.requires_grad = True
    train_loader = SalObjDataset.get_loader(args.train_root, args.train_batch//2, config.DATA.IMG_SIZE,False,args.texture,"train")

    optimizer = optim.Adam(params= model.parameters(), lr = args.lr/10, betas=[0.9,0.999], eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer,step_size = args.decay_epoch, gamma=args.gamma)
    return train_loader,optimizer,scheduler

if __name__ == '__main__':

    train_loader,val_loader,optimizer,scheduler = warmup_init(model)

    for epoch in tqdm(range(args.warmup_epoch)):
        train(train_loader, model, optimizer, scheduler, epoch)
        val(val_loader, model, epoch)
        
    train_loader,optimizer,scheduler = finetune_init(model)
    
    for epoch in tqdm(range(args.warmup_epoch,args.max_epoch)):
        train(train_loader, model, optimizer, scheduler, epoch)
        val(val_loader, model, epoch)
