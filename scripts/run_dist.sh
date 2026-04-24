
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#    --backbone pvt_v2 --lr 3e-4 --train_batch 42 --mfusion LSF \
#     --log_path ./log/ --decay_epoch 10 --gamma 0.5

CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
   --backbone segswin-base segswin-small --lr 3e-4 --train_batch 48 --mfusion LSF \
   --log_path ./log/ --decay_epoch 10 --gamma 0.5 --warmup_epoch 20