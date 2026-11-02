# Point-Query Quadtree for Crowd Counting, Localization, and More (ICCV 2023)

This repository includes the official implementation of the paper: 

[**Point-Query Quadtree for Crowd Counting, Localization, and More**](https://arxiv.org/abs/2308.13814)

International Conference on Computer Vision (ICCV), 2023

[Chengxin Liu](https://cxliu0.github.io/)<sup>1</sup>, [Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>, [Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>, [Tongliang Liu](https://tongliang-liu.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, China  

<sup>2</sup>The University of Sydney, Australia

[[Paper]](https://arxiv.org/abs/2308.13814) | [[Supplementary]](https://drive.google.com/file/d/1WxdtOaEEccYrXuNQTn1k29lFDAetBm63/view?usp=sharing)

![PET](teaser.JPG)

## Highlights

We formulate crowd counting as a decomposable point querying process, where sparse input points could split into four new points when necessary. This formulation exhibits many appealing properties:

- *Intuitive*: The input and output are both interpretable and steerable
  
- *Generic*: PET is applicable to a number of crowd-related tasks, by simply adjusting the input format
  
- *Effective*: PET reports state-of-the-art crowd counting and localization results
  

## Installation

- Required packages:
  
```
torch
torchvision
numpy
opencv-python
scipy
matplotlib
```

- Install packages:

```
pip install -r requirements.txt
```


## Data Preparation

- Download crowd-counting datasets, e.g., [ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset).
  
- We expect the directory structure to be as follows:
  

```
PET
├── data
│    ├── ShanghaiTech
├── datasets
├── models
├── ...
```

- Alternatively, you can define the path of the dataset in [datasets/__init__.py](datasets/__init__.py)

- For [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/), [JHU-Crowd++](http://www.crowd-counting.com/), and [NWPU-Crowd](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/) datasets, please refer to [preprocess_dataset.py](https://github.com/cxliu0/PET/blob/main/preprocess_dataset.py):

  * change [```dataset```](https://github.com/cxliu0/PET/blob/main/preprocess_dataset.py#L217) and [```data_root```](https://github.com/cxliu0/PET/blob/main/preprocess_dataset.py#L218)
  * run ```python preprocess_dataset.py```


## Training

- Download ImageNet pretrained [vgg16_bn](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth), and put it in ```pretrained``` folder. Or you can define your pre-trained model path in [models/backbones/vgg.py](models/backbones/vgg.py)
  

- To train PET on ShanghaiTech PartA, run
  
  ```
  sh train.sh
  ```
  

## Evaluation

- Modify [eval.sh](eval.sh)
  - change ```--resume``` to your local model path
- Run

```
sh eval.sh
```

## Pretrained Models

- Environment:
```
python==3.8
pytorch==1.12.1
torchvision==0.13.1
```

- Models:

| Dataset                  | Model Link  | Training Log  | MAE |
| ------------------------ | ----------- | --------------| ----|
| ShanghaiTech PartA       |  [SHA_model.pth](https://drive.google.com/file/d/1QwV8hrEDs1LQ4h1TH4KSL8tB51AImNMT/view?usp=drive_link)   | [SHA_log.txt](https://drive.google.com/file/d/1UpY61L0KWRA9c29CM9FMX34bHyprnPUY/view?usp=sharing) | 49.08 |
| ShanghaiTech PartB       |  [SHB_model.pth](https://drive.google.com/file/d/10HK42xC6fmOK-5lQfu-pTn6oAHYeRUhv/view?usp=sharing)   | [SHB_log.txt](https://drive.google.com/file/d/1M74PI0XuJtQraPOUiCQJSCUjrWoJUq3n/view?usp=sharing) | 6.18 |
| UCF_QNRF                 |  [UCF_QNRF.pth](https://drive.google.com/file/d/129l__gW51UtTQnPKM-90lTZo508-Eh7I/view?usp=sharing)    | - | - |
| JHU_Crowd                |  [JHU_Crowd.pth](https://drive.google.com/file/d/1D4vtoYhQuvj_5onJaXJRtWrlwrl2ckbE/view?usp=sharing)   | - | - |
| NWPU_Crowd               |  [NWPU_Crowd.pth](https://drive.google.com/file/d/1MX7tQAexyc9slrt7TaNSK7j8RtSvnI2H/view?usp=sharing)  | - | - |


## Frequently Asked Questions (FAQ)

* The model trained on my custom dataset does not perform well, why?
  * Please check the [load_data](https://github.com/cxliu0/PET/blob/main/datasets/SHA.py#L105) function in your custom dataset. The input format should be (y, x) instead of (x, y). If the input annotations are wrong during training, the output of the trained model could be abnormal.
 
* How to deal with images with no person?
  * Please refer to this [issue](https://github.com/cxliu0/PET/issues/33#issuecomment-2782560733).

## Citation

If you find this work helpful for your research, please consider citing:

```
@InProceedings{liu2023pet,
  title={Point-Query Quadtree for Crowd Counting, Localization, and More},
  author={Liu, Chengxin and Lu, Hao and Cao, Zhiguo and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```


## Permission

This code is for academic purposes only. Contact: Chengxin Liu (cx_liu@hust.edu.cn)


## Acknowledgement

We thank the authors of [DETR](https://github.com/facebookresearch/detr) and [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) for open-sourcing their work.



## 统一计数、定位与效率指标

新增 MAE、RMSE、Precision/Recall/F1@4/8/16px、逐图 CSV 与指标 JSON；
训练自动导出，参数量、延迟/FPS、峰值显存和可选 GFLOPs 由独立脚本统计。
详见 [评价说明](METRICS.md)。

PET 训练参数可加 `--loc-radii 4 8 16`；默认已启用。
独立验证示例（按实际架构保留其他参数）：
```bash
python eval.py --dataset_file Wheat1799 --data_path /path/to/RiceW3999 --resume /path/to/best_checkpoint.pth --metrics-dir evaluation/leaf
```

## 本地预训练目录
克隆仓库后已提供 `pretrained/`，将 `vgg16_bn-6c64b313.pth` 放入即可。
该文件与 P2RLoss 所需的 torchvision ImageNet VGG16-BN 权重相同；
仅骨干初始化权重可以共用，训练后的完整计数模型不能互换。
VGG 加载路径基于仓库位置解析，不再依赖启动命令时的当前目录。


## 断点续训
两个训练入口均支持每轮自动保存完整状态，并用 `--auto-resume` 自动恢复；显式恢复可用 `--resume <checkpoint.pth或实验目录>`。详见 [断点续训说明](RESUME.md)。


验证默认每个完整 epoch 执行，两个入口一致；可用 `--eval_freq N` 覆盖，最后一轮始终验证一次。完整状态每轮保存，不受验证频率影响。


## 快速数据检查（RiceW3999 / VOC 点标注）

```bash
python check_data.py --data-path data/RiceW3999                 # 默认快速检查
python check_data.py --data-path data/RiceW3999 --decode        # 完整解码，检查像素损坏
python check_data.py --data-path data/RiceW3999 --check-overlap # 完整解码并按像素查重
```

快速模式只读取图片头中的尺寸及全部标注，检查配对、完整中文/空格 ID、重复 ID、坐标有限性/边界和 train/val/test ID 重叠。
不读取全部像素，因此不能保证发现图像数据区损坏；完整检查需加 `--decode`。
`--check-overlap` 比较解码后 RGB 像素，不能排除来自同一原图的不同重叠裁块泄漏。
JSON 仍包含图片数、点数、空标注数和尺寸统计，进度写入 stderr，`--no-progress` 可关闭。
检查仅依赖 Pillow 和 Python 标准库，不导入训练模型/PyTorch；有 tqdm 时显示进度条，没有时输出计数进度。
脚本只读，不去重、不删改数据，也不会自动加入每轮训练。只适用于 JPEGImages/PointAnnotations/ImageSets 点标注目录，不替代原始 SHHA 等格式的预处理。
本次每仓库 9 项针对性数据检查测试通过；未重跑模型训练测试。

PET/P2RLoss 的检查保留重复点的计数，不会擅自删除或修改标注；这不代表重复标注必然正确。


## RiceW3999：统一实验输出

直接运行 main.py 即可同时显示终端输出并追加保存 console.log，无需额外的 logs/ 或 shell tee。
数据路径优先使用 --data_path，支持复用其他仓库的数据集。

```bash
python -u main.py \
  --dataset_file Wheat1799 \
  --data_path /root/gpufree-data/apgcc-leaf-counting/data/RiceW3999 \
  --backbone vgg16_bn --batch_size 32 --epochs 50 \
  --num_workers 4 --eval_freq 1 --seed 2024 \
  --output_dir ricew3999_pet_ep50_bs32_run1 --auto-resume
```

本例所有训练产物均位于 outputs/Wheat1799/ricew3999_pet_ep50_bs32_run1/：
- console.log：终端输出、警告及训练异常堆栈，续训追加。
- run_log.txt：结构化训练日志。
- checkpoint.pth / best_checkpoint.pth：最新完整状态与最佳模型。
- metrics_history.csv / best_metrics.json：历轮与最佳指标。
- evaluation/epoch_XXXX/：每次验证的指标和逐图结果。

已有实验目录保持兼容；旧 logs/ 文件不会自动移动或删除。
分布式运行中主进程写 console.log，其他进程写 console.rankN.log，均在同一实验目录。
此自动终端日志适用于 main.py 训练入口；独立评估仍通过 --metrics-dir 指定评估结果路径。
