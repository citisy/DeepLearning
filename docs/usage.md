# 方法使用案例

仓库把「模型怎么训、数据从哪来、分数怎么算」拆开，再在 `bundles/` 里拼成可以直接 `run()` 的类。类上的属性就是默认配置，构造时传入同名参数即可覆盖，不必改源码。

```python
from bundles.image_classification import LeNet_mnist as Process

Process(data_dir='data/mnist', device=0).run(
    max_epoch=10,
    train_batch_size=256,
    predict_batch_size=256,
)
```

`run()` 会依次完成初始化、打印模型信息、训练、把权重写到 `model_data/{model_version}/{dataset_version}/{model_name}.pth`，再在验证集上算分。训练细节通过 `fit_kwargs`、评估细节通过 `metric_kwargs` 传进去。

```python
Process().run(
    max_epoch=100,
    train_batch_size=32,
    fit_kwargs=dict(
        check_period=1,          # 每个 epoch 存一次并评估
        use_scheduler=True,
        use_ema=True,
        accumulate=64,           # 梯度累积，按样本数计
        metric_kwargs=dict(is_visualize=True, max_vis_num=8),
    ),
)
```

权重和日志在 `model_data/`，可视化结果在 `cache_data/`。版本由类属性 `model_version`、`dataset_version` 决定，也可以在构造时改掉，用来区分不同实验。

查看某个流程类支持哪些可覆盖字段：

```python
from processor import Process
print(Process.help())
```

`help()` 会扫类上带注释的属性，列出名字、类型和默认值。

## 图像分类

`bundles/image_classification.py` 里，`ClsProcess` 负责前向、Top-1 指标和可视化，`Mnist` / `Cifar` / `ImageNet` 负责读数据和增强。两者组合后只需要实现 `set_model()`。

| 类 | 数据 | 入口 |
| --- | --- | --- |
| `LeNet_mnist` | `data/mnist` | `Process().run(max_epoch=10, train_batch_size=256)` |
| `LeNet_cifar` | CIFAR-10 | `max_epoch=150, train_batch_size=128` |
| `ResNet_ImageNet` | ImageNet | 自带 ImageNet 均值方差和随机裁剪 |
| `Vit_ImageNet_Pretrained` | ImageNet | 可换成 torchvision 权重，见 `task/vit.260130.py` |

只做推理时先 `init()`，再调用 `single_predict`。`task/vit.260130.py` 的写法是：

```python
from bundles.image_classification import Vit_ImageNet_Pretrained

class Process(Vit_ImageNet_Pretrained):
    out_features = 1000

    def set_model(self):
        import torchvision
        self.model = torchvision.models.vit_b_16()

    def load_pretrained(self):
        # 覆盖默认加载逻辑，读入本地权重并设置 self.classes
        ...

process = Process()
process.init()
print(process.single_predict('out.png'))
```

`use_pretrained=True` 时，`init()` 里的 `set_model_status()` 会调用 `load_pretrained()`。

## 目标检测

`bundles/object_detection.py` 中，`OdProcess` 把图像、框和类别送进模型，验证阶段用 `metrics.object_detection.EasyMetric` 算各类 AP，返回值里的 `score` 是平均 AP。

```python
from bundles.object_detection import YoloV5_yolov5 as Process

process = Process(device=0, config_version='yolov5l', data_dir='yolov5/data_mapping')
process.run(max_epoch=21, train_batch_size=16, fit_kwargs=dict(accumulate=64))
```

`YoloV5_Voc` 走 VOC 标注，`YoloV5_yolov5` 走官方 YOLO 目录映射。输入尺寸、锚框和增强写在对应的数据混入类上，例如 `Yolov5Aug` 的 LetterBox，验证时用 `val_data_restore()` 把框映射回原图。

`FastererRCNN_Voc` 是同一套 `OdProcess` 换骨干和数据集的例子：类定义是 `OdProcess` 加 `Voc`，`set_model()` 里实例化 `models.object_detection.FasterRCNN.Model`。

## 文本分类与预训练微调

`bundles/text_classification.py` 把 BERT 流程、GLUE 数据和指标拆开再组合：

```python
from bundles.text_classification import BertHF_SST2 as Process

Process(
    pretrained_model='bert-base-uncased',
    vocab_fn='bert-base-uncased/vocab.txt',
).run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
```

- `Bert`：分类头、AdamW。
- `FromBertHFPretrained`：加载 Hugging Face 格式权重。
- `SST2` / `CoLA`：句子长度、类别数、`data_parse` 里的 Loader。
- `McMetric`：把 `metric()` 换成 Matthews 相关系数，供 CoLA 使用。

句对任务在 `bundles/text_pair_classification.py`，命名方式相同，例如 `BertHF_MNLI`、`BertHF_QQP`。

## 文本生成

`bundles/text_generation.py` 与 `bundles/text_pretrain.py` 把 Qwen2 拆成可叠加的几块：

| 混入 | 作用 |
| --- | --- |
| `BaseQwen2` | 按 `config_version`（`0.5b` / `1.5b` / `7b` / `72b`）在 `meta` 设备上建模型，避免一上来占满显存 |
| `Qwen2Trainer` | 训练步、学习率调度 |
| `Qwen2Predictor` | 生成时的 `single_predict` / `batch_predict` |
| `FromQwen2Pretrained` | 读 safetensors 或已有 checkpoint |
| `PretrainText` / `ChatText` | 预训练语料或对话数据 |
| `Qwen2TrainerWithDpo` / `Qwen2TrainerWithDistill` | 在同一套训练循环上换成 DPO 或蒸馏 |

`task/qwen2.0408.py` 是预训练脚本的典型结构：子类只改 `set_model`、`set_tokenizer` 和 `on_train_step`（这里包了一层 bfloat16 autocast），然后显式调用 `init()` 和 `fit()`，而不是 `run()`。这样可以自己决定是否在训练前加载 checkpoint、用哪一个 jsonl、按 step 还是按 epoch 做检查。

对话微调用 `Qwen2ForChatText`，DPO 用 `Qwen2ForChatTextWithDpo`，对应 `task/qwen2.chat.0408.py` 和 `task/qwen2.dpo.chat.0408.py`。

生成时：

```python
response = process.single_predict(
    text='请用 Python 写一个计算斐波那契数列的函数',
    model_kwargs=dict(max_gen_len=200),
    load_checkpoint=True,
)
print(response['text'])

responses = process.batch_predict(
    text=['为什么天空是蓝色的', '解释什么是机器学习'],
    total=2,
    model_kwargs=dict(max_gen_len=200),
    load_checkpoint=True,
)
```

`load_checkpoint=True` 会在预测开始前走已注册的加载钩子，默认读 `pretrained_checkpoint`。

## 图像生成

`bundles/image_generation.py` 里的 `SD` 同时混入了 LoRA、ControlNet、预训练加载、训练和采样：

```python
from bundles.image_generation import SD_SimpleTextImage as Process

process = Process(
    use_pretrained=True,
    pretrained_model='path/to/sd',
    vocab_fn='vocab.json',
    encoder_fn='merges.txt',
    config_version='v1.5',  # 也可为 v1 / v2 / xl
    # use_lora=True,
    # lora_pretrained_model='path/to/lora',
)
process.init()
image = process.single_predict(
    'a painting of a virus monster playing guitar',
    neg_texts='',
    is_visualize=True,
)
```

同一套 `DiProcess` 还接了 DDPM、DDIM、Flux 和 Qwen-Image。GAN 任务（WGAN、StyleGAN、pix2pix、CycleGAN）走 `GanProcess`，优化器在 `GanOptimizer` 里分开更新生成器和判别器。

## 语音与多模型流水线

单个语音模型在 `bundles/speech_recognition.py`、`speech_detection.py`、`speech_generation.py`。需要把检测、识别、标点、说话人串起来时，用 `bundles/complex_pipeline.py`。它自己也是一个 `Process`，在 `set_model()` 里创建并 `init()` 若干子流程。

```python
from bundles.complex_pipeline import FunAsr as Process

processor = Process(
    det_model_dir='.../speech_fsmn_vad_zh-cn-16k-common-pytorch',
    rec_model_dir='.../speech_paraformer-large-...',
    punc_model_dir='.../punc_ct-transformer_...',
    spk_model_dir='.../speech_campplus_sv_...',
)
processor.init()
processor.single_predict(speech_path='demo.wav')
```

OCR 同理，`PPOCRv4` / `PPOCRv6` 内部持有检测流程和识别流程：

```python
from bundles.complex_pipeline import PPOCRv4 as Process

process = Process(
    det_model_dir='.../ch_PP-OCRv4_det_server_train',
    rec_model_dir='.../ch_PP-OCRv4_rec_server_train',
    rec_processor_config=dict(vocab_fn='.../ppocr_keys_v1.txt'),
)
process.init()
process.single_predict('page.png')
```

单独训练检测或识别时，用 `bundles/object_detection.py` 的 `PPOCRv4Det_Icdar`，以及 `bundles/text_recognition.py` 的 `PPOCRv4Rec_MJSynth`。`task/` 下的脚本通常只继承这些类，改数据路径和导出方式。

## 自己接一个任务

最小写法是继承已有流程，只补模型、数据和训练步。分类任务可以沿 `ClsProcess`：

```python
from bundles.image_classification import ClsProcess
from processor import DataHooks

class MyData(DataHooks):
    dataset_version = 'my_data'
    data_dir = 'data/my_data'
    in_ch = 3
    input_size = 224
    out_features = 2

    def get_data(self, *args, train=True, **kwargs):
        # 返回 list[dict]，样本里至少要有 image 和 _class
        ...

class Process(ClsProcess, MyData):
    model_version = 'MyCls'

    def set_model(self):
        from models.image_classification.ResNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)

if __name__ == '__main__':
    Process(device=0).run(max_epoch=20, train_batch_size=32)
```

若现成的 `on_train_step` 不够用，在子类里重写它，返回值必须包含可反传的 `loss`。验证则重写 `on_val_step`（产出 `preds`）、`on_val_reprocess`（把真值和预测收进 `process_results`）和 `metric`（返回 `{模型名: {"score": ...}}`，早停和 `best.pth` 读的就是这个 `score`）。

## 分布式与超参搜索

单机多卡不用改训练步，用装饰器换 DataLoader 和设备：

```python
from processor import ddp_process_wrap

@ddp_process_wrap
class Process(ResNet_ImageNet):
    ...

# torchrun --nproc_per_node=2 train.py
```

`ds_process_wrap` 用 DeepSpeed 初始化引擎，并把 `on_backward` 换成 `model.backward` / `model.step`。配置写在类属性 `ds_config` 里。

`processor.ParamsSearch` 对 `params` 里标记为 `var` 的字段做笛卡尔积，每一组用独立的 `dataset_version` 跑一遍 `process.run()`。模型结构、学习率、学习率曲线都可以放进 `var`。用法和两组示例写在 `ParamsSearch` 的文档字符串里。
