# Deep Learning

深度学习模型基于pytorch的实现

# 说明文档

| 文档 | 内容 |
| --- | --- |
| [方法使用案例](usage.md) | 分类、检测、文本、生成、多模型串联，以及自己接一个新任务 |
| [代码运行流程](runtime.md) | `init`、`fit`、`predict`、`single_predict` 的调用链 |
| [框架设计优点](design.md) | 混入组合、钩子、检查点、分布式和超参搜索 |

代码里的可运行入口在 `bundles/`。每个任务类通常是「流程 + 数据集」的多重继承，例如 `ResNet_ImageNet(ClsProcess, ImageNet)`。

# 个人笔记

## 图像相关任务

[神经网络-CNN基础](https://www.citisy.site/posts/33979.html)

[神经网络-CNN系列](https://www.citisy.site/posts/58859.html)

[目标检测-RCNN系列](https://www.citisy.site/posts/18732.html)

[目标检测-YOLO系列](https://www.citisy.site/posts/50950.html)

[生成模型-GAN基础](https://www.citisy.site/posts/31732.html)

[生成模型-GAN系列](https://www.citisy.site/posts/64923.html)

[生成模型-Diffusion Model基础](https://www.citisy.site/posts/21195.html)

## 文本相关任务

[神经网络-RNN系列](https://www.citisy.site/posts/33259.html)

[神经网络-AE&VAE基础](https://www.citisy.site/posts/21865.html)

[神经网络-seq2seq](https://www.citisy.site/posts/51026.html)

[神经网络-transformer](https://www.citisy.site/posts/13898.html)

[预训练-BERT](https://www.citisy.site/posts/23272.html)

[预训练-GPT系列](https://www.citisy.site/posts/31506.html)

## 任务流程相关

[数学工具-常用分类任务评估指标](https://www.citisy.site/posts/20745.html)

[数学工具-常用回归任务评估指标](https://www.citisy.site/posts/28243.html)

[数学工具-常用聚类任务评估指标](https://www.citisy.site/posts/37258.html)

[数学工具-常用目标检测任务评估指标](https://www.citisy.site/posts/35178.html)

# 目录结构

```
.
├── data_parse  # 数据解析类
├── examples    # 任务使用类
├── metrics     # 模型评估类
├── models      # 模型实现类
├── processor   # 流程处理类
└── utils       # 通用工具类
```

# 运行环境

所有脚本均在`python==3.8, pytorch==1.11, torchvision==0.12` 环境下运行通过