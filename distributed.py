import os
import sys
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import json
from networks.models_config import parse_option

from datetime import datetime
import SalObjDataset
import logging
from torch.optim import lr_scheduler
from networks.GSformer import GSformer
from tqdm import tqdm
from loss import *
from utils import save_project,clip_gradient
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


CELoss = torch.nn.BCEWithLogitsLoss()
IOULoss = IOU()


step = 0

mae_list = []
miou_list = []
lr_list = []

best_mae = 0
best_epoch = 0

scaler = GradScaler()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

    # Data loading code

def print_once(*value):
    if args.local_rank == 0:
        print(*value)

def log_once(msg):
    if args.local_rank == 0:
        logging.info(msg) 

args,config = parse_option()


f = open(os.devnull, "w")
if args.local_rank != 0:
    sys.stdout = f
    sys.stderr = f

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')

cudnn.benchmark = True
args.nprocs = torch.cuda.device_count()

model = GSformer(config)
model.cuda(args.local_rank)
model = torch.nn.parallel.DistributedDataParallel(model,
                device_ids=[args.local_rank],find_unused_parameters=True)


if args.local_rank == 0:
    def log_name(backbone):
        log_prefix = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'GSformer-' 
        if len(backbone) == 1:
            return log_prefix + backbone[0]+"+"+backbone[0]
        else:
            return log_prefix + backbone[0]+"+"+backbone[1]

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    log_path = args.log_path + (args.tag if args.tag else log_name(args.backbone)) + '/'
    src_path = log_path+'src/'
    fig_path = log_path+'fig/'
    ckpt_path = log_path +'ckpt/'


    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(fig_path)
        os.makedirs(ckpt_path)
        os.makedirs(src_path)
    summaryWriter = SummaryWriter(fig_path)
    save_project(src_path,['./','./networks'])
    with open(log_path+"args.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    config.dump(stream=open(log_path+"config.yaml", "w"))
dist.barrier()
log_once("Config:")
log_once(config)
log_once("Args:")
log_once(args)

def train(train_loader, model, optimizer, epoch, local_rank, args):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0
    
    
    total_step = len(train_loader)
    
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts,depths,texs,bounds) in enumerate(train_loader, start=1):
            
            with autocast():
                optimizer.zero_grad()
                images = images.cuda(local_rank,non_blocking=True)
                depths = depths.cuda(local_rank,non_blocking=True)
                gts = gts.cuda(local_rank,non_blocking=True)
                texs = texs.cuda(local_rank,non_blocking=True)
                bounds = bounds.cuda(local_rank,non_blocking=True)

                s1, s2, s3, s4, edge1, edge2, edge3 = model(images, depths)

                loss = GTSupervision(s1,s2,s3,s4,gts) + EdgeSupervision(edge1,bounds)\
                    + NAMLABSupervision(edge2,texs) + EdgeSupervision(edge3, bounds)


            torch.distributed.barrier() 

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
                log_once('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch+1, args.max_epoch, i, total_step, loss.item()))

                if args.local_rank==0:
                    summaryWriter.add_scalars("train_loss", {"epoch_loss": loss.item()}, step)
        scheduler.step()
        loss_all /= epoch_step
        log_once('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch+1, args.max_epoch, loss_all))

        if args.local_rank == 0 and ((epoch+1) % 10 == 0 or (epoch+1) == args.max_epoch):
            torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch+1))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        #torch.save(model.state_dict(), ckpt_path + 'Epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# val function
def val(val_loader, model, epoch, local_rank, args):
    global best_mae, best_epoch
    model.eval()
    
    with torch.no_grad():
        mae_loss = torch.nn.L1Loss()
        sum_IoU = 0.0
        sum_mae = 0.0

        for  _, (images, gts,depths) in enumerate(val_loader, start=1):

            gts = gts.cuda(local_rank,non_blocking=True)
            images = images.cuda(local_rank,non_blocking=True)
            depths = depths.cuda(local_rank,non_blocking=True)

            res = model(images, depths)
            res = torch.sigmoid(res[0])

            sum_mae += mae_loss(res,gts)*len(gts)
            sum_IoU += IoU(res,gts)

        dist.barrier()
        dist.all_reduce(sum_mae,op = dist.ReduceOp.SUM)
        dist.all_reduce(sum_IoU,op = dist.ReduceOp.SUM)
        print(len(val_loader.dataset))
        mae_list.append(sum_mae.item()/len(val_loader.dataset))
        miou_list.append(sum_IoU.item()/len(val_loader.dataset))
        lr_list.append(optimizer.param_groups[0]['lr'])

        print("MIoU:",miou_list[-1])
        print("MAE:", mae_list[-1])
        print("lr:",lr_list[-1])
        if args.local_rank == 0:
            summaryWriter.add_scalars("val_loss", {"mae_loss": mae_list[-1],"miou_loss": miou_list[-1]}, epoch)

        
        if epoch == 0 and args.local_rank == 0:
            best_mae = mae_list[-1]
            torch.save(model.state_dict(), ckpt_path + "Best_mae_test.pth")
            print('update best epoch to epoch {}'.format(epoch))
        else:
            if mae_list[-1] < best_mae and args.local_rank == 0:
                best_mae = mae_list[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path + "Best_mae_test.pth")
                print('update best epoch to epoch {}'.format(epoch))
        log_once('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae_list[-1], best_epoch,best_mae ))
        print("Best MAE",best_mae)
        print("Best Epoch",best_epoch)

if __name__ == '__main__':
    """
    usage:
    TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed.py --backbone segswin-base --texture /namlab40/ --lr 3e-4 --train_batch 64 --mfusion HAIM
    """
    
    for name, para in model.named_parameters():
        if "encoderR" in name or "encoderD" in name:
            para.requires_grad_(False)
    train_loader = SalObjDataset.get_loader(args.train_root, args.train_batch, config.DATA.IMG_SIZE,True,args.texture,"train")
    val_loader = SalObjDataset.get_loader(args.val_root, args.train_batch, config.DATA.IMG_SIZE,True,args.texture,ds_type="val")
    optimizer = optim.Adam(params=model.parameters(),lr = args.lr,betas=[0.9,0.999], eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=args.decay_epoch//2,gamma=args.gamma)

    if args.local_rank == 0:
        print("board",args.local_rank)
        os.popen(f"tensorboard --logdir={fig_path} --port 24423 --bind_all")

    for epoch in tqdm(range(args.warmup_epoch)):
        train_loader.sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        val_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args.local_rank, args)
        val(val_loader, model, epoch, args.local_rank, args)\

    for name, para in model.named_parameters():
        if "encoderR" in name or "encoderD" in name:
            para.requires_grad_(True)


    torch.cuda.empty_cache()
    train_loader = SalObjDataset.get_loader(args.train_root, args.train_batch//4, config.DATA.IMG_SIZE,True,args.texture,"train")

    optimizer = optim.Adam(params= model.parameters(), lr = args.lr/20, betas=[0.9,0.999], eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer,step_size = args.decay_epoch, gamma=args.gamma)


    for epoch in tqdm(range(args.warmup_epoch,args.max_epoch)):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args.local_rank, args)
        torch.cuda.empty_cache()
        val(val_loader, model, epoch, args.local_rank, args)