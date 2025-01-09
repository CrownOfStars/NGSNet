# NGSNet

This is an official implementation for "NGSNet: An NAMLab Hierarchical Image Segmentation Guided Swin Transformer Based Network for RGB-D Salient Object Detection"

## Environmental Setups

python>=3.7 pytorch>=1.13

```
conda create -n ngsnet python=3.8
conda activate ngsnet
pip install -r requirements.txt
```

>## Evaluation and Visual Analysis

>## Results
* **Qualitative comparison**  

![](./ngs_table.png)
![](./pr_curve.png)

Fig.1 Qualitative comparison of our proposed method with some RGB-D SOTA methods.  

![](./rgbt_table.png)

Fig.2 Qualitative comparison of our proposed method with some RGB-T SOTA methods.

* **Quantitative comparison** 

![](./main_cmp.png)

Table.1 Quantitative comparison with some SOTA models on some public RGB-D benchmark datasets. 

![](./rgbt.png)

Table.2 Quantitative comparison with some SOTA models on some public RGB-D benchmark datasets. 


* **Salmaps**   

The salmaps of the above datasets can be download from [here]().

>## Train/Test

>## Data Preparation

We provide [download link](https://pan.baidu.com/s/1dZ47EX1ttETSE3jF8Km-5w&pwd=yial) for the RGB-D dataset，[download link](https://pan.baidu.com/s/1dZ47EX1ttETSE3jF8Km-5w&pwd=yial) for the RGB-T dataset.

We randomly selected images from multiple test datasets for validation.

### Dataset Structure

```
dataset/
├─RGBD_dataset/
│ ├─train/
│ │ ├─ReDWeb-S-TR/
│ │ ├─NJUNLPR/
│ │ ├─...
│ └─test/
│   ├─NJU2K/
│   ├─STERE/
│   ├─...
└─RGBT_dataset/
  ├─train/
  │ └─RGBT_train/
  └─test/
    ├─VT821/
    ├─VI-RGBT1500/
    ├─...
```
The structure of each dataset is shown below
```
RGBT_train/
├─bound/
├─GT/
├─T/
├─RGB/
├─namlab40/ #only for train, optional
├─...
```

>### NAMLab boundary guidance mechaism

>#### config NAMLAB environment

refer to [opencv-install](https://waltpeter.github.io/open-cv-basic/install-opencv-ubuntu-cpp/index.html)

refer to [matlab-install](https://blog.csdn.net/mziing/article/details/122422397)

>#### NAMLab boundary data preparation

refer to [NAMLab](https://github.com/YunpingZheng/NAMLab)

>### convert NAMLab Hierarchical Image Segmentation map to NAMLab boundary map

```
python convertmat.py --/path/to/data --/path/to/result
```

>### pretrain

./pretrained contains several backbone pre-trained checkpoint files with their corresponding configuration files

train on multi-GPUs

```
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed.py \
#     --backbone segswin-base segswin-small --texture /namlab40/ --lr 3e-4  --decay_epoch 10 --gamma 0.5 \
#     --train_batch 32 --mfusion AFM  --warmup_epoch 40 --max_epoch 100 \
#     --train_root /path/to/train/dataset --val_root /path/to/test/dataset
```

test 
```
python test.py --test_model /path/to/log/ --gpu_id 0
```

>### model and log

log文件夹里面是模型运行的结果，这个文件夹不要动，别的文件丢了都无所谓，这里面的文件都是很重要的数据

每次运行后会创建一个文件夹，里面包含如下几组数据

+ ckpt文件夹:包含了模型文件，一般取best那个pth模型文件

+ src文件夹:本次运行的源文件，这样修改了代码也不用担心之前的代码没存档了(我认为是不错的习惯，因为代码在初期是经常修改的，把代码和模型运行结果对应起来方便还原)

+ save文件夹:在测试集上生成的显著性检测的结果

+ log.txt文件:输出日志，除了输出外，本次运行的config信息可以从里面得知



