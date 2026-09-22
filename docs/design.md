# 框架设计优点

这套代码要同时覆盖分类、检测、分割、OCR、语音、语言模型和扩散模型。设计上的取舍是：把「循环怎么转」固定在 `processor.Process`，把「这一步算什么」留给多重继承拼出来的任务类。换任务时改的是混入，而不是再写一套 Trainer。

## 流程与任务正交

`Process` 的五个基类各管一件事，任务类不继承一个巨大的训练器，而是按需要把小块叠上去。

图像分类里，`ResNet_ImageNet(ClsProcess, ImageNet)` 表示：前向和 Top-1 指标来自 `ClsProcess`，ImageNet 的路径、输入尺寸和增强来自 `ImageNet`。文本分类里，`BertHF_CoLA(McMetric, Bert, FromBertHFPretrained, CoLA)` 把 Matthews 相关系数、分类头、HF 权重和 CoLA 数据分成四块。DPO 也是同一方式：`Qwen2ForChatTextWithDpo` 在对话模型上再混入 `Qwen2TrainerWithDpo`，训练循环仍是原来的 `fit()`。

这样做的直接结果是：

- 换数据集通常只换一个 DataHooks 子类，`on_train_step` 可以留着。
- 换指标只覆盖 `metric()`，或像 `McMetric` 那样做成可复用混入。
- LoRA、ControlNet、预训练加载都是可选混入。`SD` 同时带上 `WithSDLora` 和 `WithSDControlNet`，不用时不进计算图。
- 检测、识别各自能训练，`complex_pipeline.PPOCRv4` 再把两个已经 `init()` 过的流程串成端到端推理。语音的 `FunAsr` 同样把 VAD、识别、标点、说话人放成四个子 `Process`。

多模型流水线没有另起一套运行时。子流程和单模型共用 `init`、设备、权重加载和 `single_predict`。

## 钩子而不是回调迷宫

训练、验证、推理各有一条写死的顺序，扩展点是步骤函数和四个容器。

步骤函数（`on_train_step`、`on_val_step`、`gen_predict_inputs`）表达任务差异，签名稳定：训练步返回带 `loss` 的字典，`metric()` 返回带 `score` 的字典。早停、`best.pth` 和日志只认这两个键，所以新任务接进循环时不必改 `fit()`。

容器（`register_train_start` 等）表达与任务无关的副作用：wandb、额外 checkpoint、DDP 包装、验证后清缓存。`ddp_process_wrap` 和 `ds_process_wrap` 是这一设计的结果。装饰器生成子类，替换 DataLoader 和 `on_backward`，原任务类的 `on_train_step` 保持不动。单卡脚本和 `torchrun` / `deepspeed` 启动的脚本可以是同一个类。

`fit()` 的参数会传到每一步。`batch_size`、`check_period`、`model_kwargs` 不需要在 Process 上再包一层配置对象，子类按名字接收即可。本次调用的参数会落到 `work_dir/train_kwargs.yml`，实验目录里能直接看到当时怎么跑的。

## 配置写在类属性上

可调字段是类属性，而不是散落的 yaml 层级。`input_size`、`data_dir`、`max_seq_len`、`config_version` 都这样声明，并可以用 `Annotated` 写注释。构造时 `self.__dict__.update(kwargs)` 覆盖同名属性，因此：

```python
Process(input_size=512, device=1, dataset_version='exp1')
```

只影响这一次实例。`Process.help()` 能把这些字段扫出来，作为该类的参数说明。

模型结构用另一层小配置。Qwen2、YOLO、Stable Diffusion 通过 `config_version` 选择预设，再用 `configs.ConfigObjParse.merge_dict` 把调用方的 `model_configs` 叠上去。大模型在 `meta` 设备上构建，等到 `fit(init_weight=True)` 或加载 checkpoint 时才落到真实显存，避免「先分配一整份随机权重，再立刻被预训练权重覆盖」。

`ParamsSearch` 复用同一套实例化方式。它把 `var` 字段做笛卡尔积，每一组新开一个 `dataset_version` 并调用 `run()`。搜索对象就是普通的 Process 子类，不需要为了调参再包一层实验管理器。

## 数据接口保持为字典

`data_parse` 的 Loader 输出 `list[dict]`，增强器 `Apply` / `RandomApply` 接收并返回同一个字典，Dataset 的 `__getitem__` 也返回字典。模型输入在 `get_model_inputs()` 才变成张量。

统一成字典之后，图像翻转、检测框同步变换、文本拼接可以写成小函数再 `Apply([...])` 串起来，而不必为每种模态定义 Transform 类型。验证阶段的 `val_data_restore()` 用同一份字典把尺度变回去，训练增强和评估坐标不会各写一套。

Dataset 形态按数据规模替换，不改循环：

| 类 | 适用情况 |
| --- | --- |
| `BaseDataset` / `BaseImgDataset` | 可随机访问的中小数据集，图像在取样本时再读 |
| `IterDataset` / `BatchIterDataset` | 流式样本，按批产出 |
| `MixDataset` | 多数据源按索引混合 |
| `IterRedisDataset` | 样本缓存在 Redis，训练进程只消费 |

`collate_fn` 挂在 Dataset 上。检测框长度不一致时，由数据集自己决定如何组 batch，DataLoader 构造代码保持通用。

## 评估和可视化与权重更新分开

`metrics/` 不参与反传。分类有混淆矩阵和 P/R/F，检测有 IoU 与 AP，分割有 mask IoU，OCR 和文本生成有各自的对齐方式。任务的 `metric()` 只负责把 `predict()` 的 `trues` / `preds` 送进去，并挑一个数作为 `score`。

因此同一套检测指标可以给 Faster R-CNN 和 YOLO 用，换模型时不用复制 AP 计算。可视化走 `visualize()`，由 `is_visualize` 和 `max_vis_num` 控制，默认把图写到 `cache_dir`，并在 wandb trace 里留一份。关掉可视化不会改变分数。

`models` 字典让 EMA 权重和原始权重在同一次验证里各算各的分数，调用方可以对比，而不必在脚本里手写第二套预测循环。

## 检查点格式跟着部署走

训练中的周期保存默认是 state dict。需要离开 PyTorch 时，同一对象上的 `save()` 还能导出 safetensors、TorchScript、ONNX 和 Triton。`include` / `exclude` 用来在加载时跳过不匹配的头，例如换分类数之后保留骨干。

附加状态用注册函数挂上，而不是把所有可选项写进 `save_weight()`。EMA 文件、wandb run id 都是这样接进去的。新的副作用增加时，保存主路径不用再加参数。

## 日志是可替换的后端

`LogHooks` 用注册表管理后端，默认是标准 logging，可选 loguru、wandb、tensorboard。`trace()` 先把指标放进缓冲区，到检查周期再 `log_trace()` 一次刷出，避免每个 step 都对远程后端发请求。进度条通过 `register_logger('pbar', pbar.set_postfix)` 接进同一个 `log()`，训练步末尾的 loss 和学习率不用单独打印。

`use_wandb=False` 或未安装 wandb 时，后端换成空实现，任务代码里的 `self.trace(..., WANDB)` 仍然安全。

## 适用边界

这套拆分适合「同一训练循环、多种任务」的实验仓库：新增一个模型，优先找最接近的 `bundles` 类继承，只覆盖 `set_model`、`get_data` 和必要的 step。它不替代通用训练框架里的自动分布式规划或配置编译。DeepSpeed 的 ZeRO、张量并行需要自己填 `ds_config`；数据路径、词表和预训练权重在 `task/` 脚本里写明，仓库没有一份全局配置把所有实验收成同一个命令。

根目录 README 中的环境记录是 Python 3.8、PyTorch 1.11、torchvision 0.12。当前生成和语言模型脚本使用了 `torch.amp` 以及可选的 DeepSpeed，这些任务需要与之匹配的较新 PyTorch。
