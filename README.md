# Test_NAMSwin

### 项目结构简介

```
#TODO 还有一些目录需要完善，目前只列出前两级目录

./WaveNet
├── cppsrc
│   └── NAMLab
├── dataset
│   ├── bd_process.py
│   ├── checkpoints
│   ├── convertmat.py
│   ├── dataset.py
│   ├── main.py
│   ├── RGBD_dataset
│   ├── ted.py
│   └── utils
├── depthcatbound.py
├── log
│   ├── 2023-12-10-10:09:57AnyNet-pool-s36
│   ├── 2023-12-10-10:12:12AnyNet-swin-large
│   ├── 2023-12-10-14:12:50AnyNet-swin-base
│   └── ...
├── loss.py
├── matlab_eval
│   ├── datasets
│   ├── EvaluationCode
│   ├── result
│   └── test_maps
├── meter.py
├── networks
│   ├── AnyNet.py
│   ├── configs
│   ├── models_config.py
│   ├── PoolNets.py
│   ├── PWNet.py
│   ├── swin_mlp.py
│   ├── SwinNet
│   ├── SwinNets.py
│   ├── swin_transformer_moe.py
│   ├── swin_transformer.py
│   ├── swin_transformer_v2.py
│   ├── wavemlp.py
│   └── Wavenet.py
├── pretrained
│   ├── configs
│   ├── poolformer_m36.pth
│   ├── swin_base_patch4_window12_384_22k.pth
│   ├── WaveMLP_M.pth
│   ├── ...
├── README.md
├── rgbd_dataset.py
├── rgbdt_dataset.py
├── run_model.py
├── test.py
├── train.py
└── utils.py

```

>### NAMLab边界提取

>#### NAMLAB环境配置

[参考教程](https://waltpeter.github.io/open-cv-basic/install-opencv-ubuntu-cpp/index.html)

```

```

```

```

>#### NAMLab分块

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

>### pytorch环境配置

```
conda create -n env_name python=3.7
conda activate env_name
pip install -r requirements.txt
```

>### 数据集设置

```
cd /home/data1/ShiqiangShu/WaveNet/dataset/
```

这个文件夹主要就是存放数据集和一些数据处理的文件

test,train,val分别存放三类数据集

我们提供了数据集的[百度网盘下载链接](https://pan.baidu.com/s/1dZ47EX1ttETSE3jF8Km-5w&pwd=yial)

```
RGBD_dataset
├── test
│   ├── COME-E
│   ├── COME-H
│   ├── DES
│   ├── DUT
│   ├── LFSD
│   ├── NJU2K
│   ├── NLPR
│   ├── SIP
│   ├── SSD
│   └── STERE
├── train
│   ├── COME
│   ├── DUT-RGBD
│   └── NJUNLPR
└── val
    └── NJUNLPR
```

测试和对比实验主要就是围绕着test里的几个数据集来展开的

>### 模型和日志

log文件夹里面是模型运行的结果，这个文件夹不要动，别的文件丢了都无所谓，这里面的文件都是很重要的数据

每次运行后会创建一个文件夹，里面包含如下几组数据

+ ckpt文件夹:包含了模型文件，一般取best那个pth模型文件

+ fig文件夹:暂时没啥用，后面可能要做loss随着epoch变化图？

+ src文件夹:本次运行的源文件，这样修改了代码也不用担心之前的代码没存档了(我认为是不错的习惯，因为代码在初期是经常修改的，把代码和模型运行结果对应起来方便还原)

+ save文件夹:在测试集上生成的显著性检测的结果

+ log.txt文件:输出日志，除了输出外，本次运行的config信息可以从里面得知

>### 评估与可视化结果

```
./matlab_eval/EvaluationCode
```

里面有评估的代码,主要是评估S-measure,F-measure,E-measure,MAE这些指标

显著性检测的结果放在test_maps,数据集放datasets，具体路径参考我的

在main.m中修改路径和模型名称,文件名称之后，在命令行也可以执行，执行速度很慢，所以可以开多个控制台同时运行

```
#usage:
matlab -nodesktop -nosplash -r "Models={'Model Name'};Datasets={'Test Dataset Name'}; main"
```

```
./matlab_eval/
├── best_model_count.png
├── check_shape.py
├── cmd.txt
├── datasets
├── EvaluationCode
├── EvaluationCode.zip
├── result
├── result2csv.py
├── subset_result
├── table_adp.csv
├── table_adp_em.csv
├── table_adp_fm.csv
├── table_adp_mae.csv
├── table_adp_sm.csv
├── table_max.csv
├── table_mean.csv
├── test_maps
└── val_result.py
```

运行后的结果将保存于./matlab_eval/result中,执行以下命令让结果转化为csv表格

```
python result2csv.py
```

>### 预训练backbone模型

pretrained包含几种backbone的预训练模型文件与之对应的配置文件

这里我已经配置好了，不需要去修改了

>### 模型和训练代码

networks里面是模型的代码

train.py的使用方法

```
python train.py --backbone {} --texture {} --gpu_id {}
```

test.py的使用方法

```
python test.py --backbone {} --test_model {} --gpu_id {}
```

其他参数的修改见

```
# model_config.py
cd /home/data1/ShiqiangShu/WaveNet/networks/
```
