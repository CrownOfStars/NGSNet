# NGSNet

This is an official implementation for "NGSNet: An NAMLab Hierarchical Image Segmentation Guided Swin Transformer Based Network for RGB-D Salient Object Detection"

## Environmental Setups

python>=3.7 pytorch>=1.13

```
conda create -n ngsnet python=3.8
conda activate ngsnet
pip install -r requirements.txt
```

>### 模型和日志

log文件夹里面是模型运行的结果，这个文件夹不要动，别的文件丢了都无所谓，这里面的文件都是很重要的数据

每次运行后会创建一个文件夹，里面包含如下几组数据

+ ckpt文件夹:包含了模型文件，一般取best那个pth模型文件

+ src文件夹:本次运行的源文件，这样修改了代码也不用担心之前的代码没存档了(我认为是不错的习惯，因为代码在初期是经常修改的，把代码和模型运行结果对应起来方便还原)

+ save文件夹:在测试集上生成的显著性检测的结果

+ log.txt文件:输出日志，除了输出外，本次运行的config信息可以从里面得知

>## Evaluation and Visual Analysis

>## Results
* **Qualitative comparison**  

![image](figs/vision_results.png)  
Figure.2 Qualitative comparison of our proposed method with some SOTA methods.  

* **Quantitative comparison** 

![image](figs/qulities_results.png)  
Table.1 Quantitative comparison with some SOTA models on five public RGB-D benchmark datasets. 

* **Salmaps**   
The salmaps of the above datasets can be download from [here](https://pan.baidu.com/s/1sswZiW-2lDaYPPnpK9Ahbw) [code:NEPU] or [Google](https://drive.google.com/file/d/1cBSijVa52ut-htYnBWDegFiYlMUhZC8W/view?usp=drive_link).


>## Train/Test

networks里面是模型的代码

>## Data Preparation

We provide [download link](https://pan.baidu.com/s/1dZ47EX1ttETSE3jF8Km-5w&pwd=yial) for the RGB-D dataset，[download link](https://pan.baidu.com/s/1dZ47EX1ttETSE3jF8Km-5w&pwd=yial) for the RGB-T dataset

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

>### NAMLab边界提取

>#### NAMLAB环境配置

[参考教程](https://waltpeter.github.io/open-cv-basic/install-opencv-ubuntu-cpp/index.html)

```

```

>#### NAMLab boundary data preparation

cppsrc文件夹包含了NAMLab的C++ 代码，详情见[NAMLab](https://github.com/YunpingZheng/NAMLab)


run_dataset,run_demo都已经编译完成

```
cd /home/data1/ShiqiangShu/WaveNet/cppsrc/NAMLab/NAMLab/'Source Codes'/Linux
```

只需要在这个目录下执行

```
run_dataset {输入图片所在文件夹路径，最好用绝对路径} {结果所在的路径，最好也是绝对路径} 0 {分割块的数量}
```

在结果的文件夹下面找到分割块的.mat文件

>### NAMLab分块文件转化为边界图

把.mat文件中每次分割出来的块边界做个提取,拿形态学膨胀的结果减去形态学腐蚀的结果

[参考博客](https://blog.csdn.net/wangjia2575525474/article/details/117919453)


```
#TODO 参数
python /home/data1/ShiqiangShu/WaveNet/dataset/convertmat.py --/path/to/data --/path/to/result
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



