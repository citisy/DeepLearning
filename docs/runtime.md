# 代码运行流程

一次实验的对象是 `processor.Process`。它本身几乎不写具体任务，只把五组钩子按固定顺序串起来：

```text
LogHooks          日志、wandb / tensorboard
DataHooks         读数据、增强、collate、组装模型输入
CheckpointHooks   保存 / 加载权重、safetensors、TorchScript、ONNX、Triton
ModelHooks        训练循环、验证循环、推理循环、优化器与早停
ApiHooks          可选的对外接口
```

`bundles/` 里的任务类继承 `Process`，再混入数据集和指标。`task/` 里的脚本再继承这些任务类，填入本机路径后调用 `init`、`fit` 或 `run`。

## 目录各自做什么

| 目录 | 在流程中的位置 |
| --- | --- |
| `data_parse/` | `Loader` 把磁盘数据变成 `list[dict]`；`Apply` / `RandomApply` 做增强；`DataRegister` 区分 train / val / test 以及图像是路径还是数组 |
| `models/` | `nn.Module`。`forward` 在训练时返回带 `loss` 的字典，在验证时返回 logits、框或生成结果 |
| `bundles/` | 把模型、数据和 `on_train_step` / `metric` 绑成一个类 |
| `processor/` | 循环、钩子注册、检查点、DataLoader 构造 |
| `metrics/` | 分类、检测、分割、OCR、文本生成的指标，由各任务的 `metric()` 调用 |
| `utils/` | 设备、初始化、EMA、早停、配置合并、日志 |

样本在流程里一直是字典。图像任务常见键是 `image`、`_class` 或 `bboxes` / `classes`；文本任务常见键是 `text`、`input_ids`。增强函数就地 `update` 这个字典，而不是换成另一种 Dataset 接口。

## 初始化

`run()` 和手写脚本都会先走到 `init()`：

```text
init()
  ├─ init_logs()     按 default_logger_types 注册 logging，以及 wandb（仅当 use_wandb=True）
  ├─ init_paths()    work_dir、cache_dir、default_model_path
  └─ init_components()
        ├─ setup_seed()
        ├─ set_device()          None → cuda；整数 → cuda:{id}；无 GPU → cpu
        ├─ set_tokenizer()       文本 / 多模态任务在这里建分词器
        ├─ set_counter()         epoch、step、检查周期计数
        ├─ set_model()           子类必须实现；大模型常用 torch.device('meta') 先占位
        ├─ models[model_name] = model
        └─ set_model_status()    use_pretrained 时 load_pretrained()，然后 model.to(device)
```

`fit()` 和 `predict()` 开头都会检查 `self.models` 是否为空。空则说明还没有 `init()`。

路径规则：

```text
work_dir  = { _model_cache_dir }/{ model_version }/{ dataset_version }
cache_dir = { _result_cache_dir }/{ model_version }/{ dataset_version }
默认权重  = { work_dir }/{ model_name }.pth
```

`_model_cache_dir` 默认 `model_data`，`_result_cache_dir` 默认 `cache_data`。

## 一条龙：`run()`

```text
run(max_epoch, train_batch_size, predict_batch_size, fit_kwargs, metric_kwargs)
  ├─ init()
  ├─ model_info()          打印参数量等概况
  ├─ fit(...)              训练，期间按周期调用 metric()
  ├─ save(default_model_path, save_type=WEIGHT)
  └─ metric(...)           训练结束后再评一次，并把分数打到日志
```

需要分步控制时可以不调用 `run()`，像 `task/qwen2.0408.py` 那样只调用 `init()` 和 `fit()`。

## 训练：`fit()`

`fit()` 先把本次参数写成 `work_dir/train_kwargs.yml`，再进入循环。传入的关键字会原样传到各个 `on_train_*`，所以 `batch_size`、`check_period`、`model_kwargs` 可以在任意一步里按名字取用。

```text
fit()
  └─ on_train_start()
        ├─ 必要时 to_empty + initialize_layers（meta 权重或 init_weight=True）
        ├─ get_train_dataloader()
        ├─ 若 is_metric：get_val_dataloader()
        ├─ use_optimizer / use_ema / use_early_stopper / use_scaler / use_scheduler
        │    为真且对应对象还不存在时，调用 set_*()
        └─ 执行 train_start_container（加载 checkpoint、初始化 wandb、DDP 包装等）
  └─ on_train()
        └─ for epoch
              on_train_epoch_start()
              for batch
                on_train_step_start()
                on_train_step()          子类实现，返回含 loss 的 dict
                on_backward()            loss.backward、累积、scaler、EMA
                on_train_step_end()      记 loss / lr；step 策略下调度和检查
              on_train_epoch_end()       epoch 策略下调度和检查
  └─ on_train_end()        跑 train_end_container，丢掉 optimizer 等临时对象
```

`on_train_step()` 的约定很窄：返回字典里要有名为 `loss` 的张量，名字以 `loss` 开头的其他项会被记进日志。检测、分类、语言模型都遵守这一约定，循环本身不用知道任务类型。

### 数据怎么进 `on_train_step`

```text
get_train_dataloader()
  ├─ get_train_data()  →  get_data(train=True)
  │     通常调用 data_parse 的 Loader，得到 list[dict] 或 Dataset
  ├─ train_data_preprocess()
  └─ 若还不是 Dataset，用 train_dataset_ins 包一层
        __getitem__ 里调用 train_data_augment()
        DataLoader 使用数据集的 collate_fn
```

默认 `train_dataset_ins` 是 `BaseImgDataset`：按索引取样本、可选读图、跑增强。大语料可以用 `IterDataset`、`BatchIterDataset`、`IterRedisDataset`，在对应任务里把 `train_dataset_ins` 换掉即可，`fit()` 不用改。

一个 batch 进模型前还会经过 `get_model_inputs()`。分类任务在这里把图像堆成 `x`、标签堆成 `true_label` 并搬到 `device`。检测任务额外给出 `gt_boxes` 和 `gt_cls`。

### 检查周期里发生什么

`check_strategy` 为 `epoch` 时在 `on_train_epoch_end` 检查，为 `step` 时在 `on_train_step_end` 检查。`check_period` 是间隔。命中周期、或已经是最后一个 epoch 时：

```text
_check_train()
  ├─ 把本周期平均 loss 写入 trace
  ├─ loss 为 nan / inf 时置 end_flag，循环退出
  └─ 按 max_save_weight_num 保存
        None  → last.pth（评估开启时还会有 best.pth）
        0     → 不保存
        >0    → {周期号}.pth，并限制保留个数

_check_metric()（is_metric=True）
  ├─ metric() → 内部通常是 predict() + metrics/*
  ├─ 记录 val_score/{模型名}.score
  ├─ 分数优于历史最好时另存 best
  └─ EarlyStopping 决定是否置 end_flag
```

早停默认开启（`use_early_stopper=True`）。它比较的是 `metric()` 返回值里当前 `model_name` 的 `score`。

优化器默认 Adam，`use_optimizer` 默认为真。学习率调度：`scheduler_strategy='epoch'` 用余弦 `LambdaLR`，`'step'` 用线性 warmup 调度，步数是 `max_epoch * len(dataloader)`。

## 验证：`metric()` → `predict()`

`metric()` 由各任务实现，但骨架一致：无梯度跑完验证集，再用 `metrics/` 里的函数把真值和预测变成 `score`。

```text
predict()                         @torch.no_grad
  └─ on_val_start()
        ├─ get_val_dataloader()   使用 val_data_augment，默认不 shuffle
        ├─ set_mode(train=False)
        └─ val_start_container    例如加载权重，但排除 optimizer
  └─ for batch
        on_val_step_start()
        on_val_step()             对 self.models 里每个模型前向
        on_val_reprocess()        累积 trues / preds 到 process_results
        on_val_step_end()         is_visualize 时调用 visualize()
  └─ on_val_end()                 val_end_container，返回 process_results
```

`self.models` 在开启 EMA 时会多一个 `ema`。验证步对字典里的每个模型都跑一遍，指标结果按模型名分开，日志里主要看 `model_name` 那一项。

`on_val_step` 与 `on_train_step` 的差别是 `get_model_inputs(..., train=False)` 不再塞标签，并且要把输出收成可序列化的预测。检测任务会在这里调用 `val_data_restore()`，把 LetterBox 之后的框还原到原图。

## 在线推理：`single_predict` / `batch_predict`

验证集推理走 DataLoader。对若干张图或若干段文本，走另一条几乎平行的链，这样不必先落成数据集：

```text
single_predict(*obj)  →  batch_predict([[obj], ...])，关掉进度条，取第一条
batch_predict(*objs, total, batch_size=16)
  └─ on_predict_start()          复用 val_start_container，set_mode(False)
  └─ for 每批
        gen_predict_inputs()     子类把路径、文本收成 list[dict]
        on_predict_step_start()  默认套 predict_data_augment（即验证增强）
        on_predict_step()        默认转到 on_val_step
        on_predict_reprocess()   按 return_keys 收集结果
        on_predict_step_end()    默认可视化
  └─ on_predict_end()
```

因此分类任务只要实现 `gen_predict_inputs()`（把图片路径读成 `image`），`single_predict('a.jpg')` 就会复用验证时的前处理和后处理。文本生成在 `gen_predict_inputs` 里做分词，在 `on_predict_reprocess` 里把 token 解回字符串。

`fragment_predict()` 预留给大图切块再拼回，基类未实现，由具体任务按需覆盖。

## 检查点

`save(path, save_type)` / `load(path, save_type)` 按类型分发：

| `save_type` | 常量 | 行为 |
| --- | --- | --- |
| 整模型 | `MODEL` | `torch.save` 模型对象，可附带额外字段 |
| 权重 | `WEIGHT` | `state_dict`，可用 `include` / `exclude` 过滤键 |
| Safetensors | `SAFETENSORS` | 给 Hugging Face 权重用 |
| TorchScript | `JIT` | trace 后保存 |
| ONNX | `ONNX` | trace 后导出 |
| Triton | `TRITON` | 导出服务端模型，具体格式由子类的 `save_triton` 决定 |

周期保存走 `save_pretrained_checkpoint()`。EMA、wandb id 等附加状态通过 `register_save_checkpoint` / `register_load_checkpoint_weight` 挂到同一次保存和加载上，不必改 `fit()`。

`load_checkpoint=True` 时，真正的读取发生在 `train_start_container` / `val_start_container` 里注册的函数，而不是在 `fit()` 本体中。验证加载会排除 `optimizer`，避免用训练中间状态做纯评估时出错。

## 钩子容器

`register_train_start`、`register_train_end`、`register_val_start`、`register_val_end` 把函数插进对应列表，可用 `insert_idx` 控制顺序。框架自己用它做四件事：

- 训练开始前按 flag 加载 checkpoint
- `use_wandb=True` 时在训练开始 `wandb.init`、结束时 `finish`
- 验证结束后 `torch_gc()`
- `ddp_process_wrap` / `ds_process_wrap` 在 start 钩子里把 `self.model` 包成 DDP 或 DeepSpeed engine

自定义逻辑如果只依赖「训练已经要开始」或「验证已经结束」，优先注册钩子，而不是复制 `fit()`。

## 一次分类 step 的数据形态

以 `ClsProcess.on_train_step` 为例，单个 batch 的形态变化是：

```text
DataLoader 取出 list[dict]
  每个 dict 已经过 data_augment：image 为 CHW 的 numpy，_class 为类别下标
      ↓
get_model_inputs()
  x: FloatTensor[B, C, H, W]  on device
  true_label: Tensor[B]       on device
      ↓
model(**inputs) → {'logits': ..., 'loss': ...}
      ↓
on_backward() 只取 loss
on_train_step_end() 把所有 loss* 项写入进度条
```

检测、识别、生成的差别集中在 `get_model_inputs`、`on_train_step` 和 `on_val_step` 这三个函数里。循环、保存、早停和日志保持同一条路径。
