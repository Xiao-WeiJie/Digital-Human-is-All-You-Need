# 📦 数据集准备指南 — PsyMind 系统

本文档详细说明每个数据集的获取方式、下载步骤、存放路径以及预处理方法。

---

## 目标目录结构

全部原始数据统一放在项目根目录的 `datasets/` 下：

```
psymind/
├── datasets/                          ← 所有原始数据放这里
│   ├── psyqa/                         ← 优先级一：PsyQA 数据集
│   │   ├── train.json
│   │   ├── valid.json
│   │   └── test.json
│   │
│   ├── mhgap/                         ← 优先级一：WHO mhGAP-IG 指南
│   │   └── mhgap-ig-v2.pdf
│   │
│   ├── openstax_psychology/           ← 优先级一：OpenStax 心理学教材
│   │   └── Psychology2e-WEB.pdf
│   │
│   ├── cpsycoun/                      ← 优先级二：CPsyCoun 对话数据集
│   │   └── CPsyCounD.json
│   │
│   └── guardrails/                    ← 优先级三：量表科普说明（自行编写）
│       ├── phq9_intro.txt
│       └── gad7_intro.txt
│
├── data/                              ← 代码中的数据加载模块
├── config/
├── ...
└── main.py
```

---

## 一、优先级一：科普核心库

### 1.1 PsyQA 数据集（清华大学 thu-coai）

| 项目 | 内容 |
|------|------|
| **论文** | *PsyQA: A Chinese Dataset for Generating Long Counseling Text for Mental Health Support* (ACL 2021) |
| **规模** | 约 22,000 条中文心理问答对，回答平均约 500 字 |
| **格式** | JSON |
| **许可** | 需签署用户协议 |

#### 获取方式（二选一）

**方式 A：通过 HuggingFace 镜像直接下载（推荐，已有 train/valid/test 划分）**

```bash
# 安装依赖
pip install datasets huggingface_hub

# Python 方式下载
python -c "
from datasets import load_dataset
ds = load_dataset('lsy641/PsyQA')
ds['train'].to_json('datasets/psyqa/train.json', force_ascii=False)
ds['validation'].to_json('datasets/psyqa/valid.json', force_ascii=False)
ds['test'].to_json('datasets/psyqa/test.json', force_ascii=False)
print(f'下载完成: train={len(ds[\"train\"])}, valid={len(ds[\"validation\"])}, test={len(ds[\"test\"])}')
"
```

或者使用 `huggingface-cli`：

```bash
# 整个数据集克隆
mkdir -p datasets/psyqa
huggingface-cli download lsy641/PsyQA --repo-type dataset --local-dir datasets/psyqa
```

**方式 B：向原作者申请完整数据集**

1. 访问官方仓库：https://github.com/thu-coai/PsyQA
2. 下载仓库中的《PsyQA数据集使用用户协议》PDF
3. 填写用户信息、授权时间、电子签名
4. 将签好的 PDF 发送至：`h-sun20@mails.tsinghua.edu.cn`
5. 审核通过后会收到含 `PsyQA_full.json` 的下载链接
6. 下载后放入 `datasets/psyqa/` 目录

#### 数据格式示例

```json
{
  "question": "最近总是很焦虑怎么办？感觉压力很大...",
  "answer": "焦虑是一种很常见的情绪体验，尤其是在面对压力的时候...",
  "topic": "焦虑",
  "strategy": ["自我暴露", "提供建议"]
}
```

---

### 1.2 WHO mhGAP-IG 干预指南（世界卫生组织）

| 项目 | 内容 |
|------|------|
| **全称** | mhGAP Intervention Guide for Mental, Neurological and Substance Use Disorders, Version 2.0 |
| **语言** | 英文（暂无官方中文版） |
| **格式** | PDF, 约 170 页 |
| **许可** | CC BY-NC-SA 3.0 IGO，免费获取 |

#### 下载步骤

```bash
mkdir -p datasets/mhgap

# 从 WHO IRIS 官方下载（约 5MB）
wget -O datasets/mhgap/mhgap-ig-v2.pdf \
  "https://iris.who.int/bitstream/handle/10665/250239/9789241549790-eng.pdf?sequence=1"
```

如果 `wget` 下载失败，手动下载：
1. 访问：https://www.who.int/publications/i/item/9789241549790
2. 点击页面上的 "Download (PDF)" 按钮
3. 保存到 `datasets/mhgap/mhgap-ig-v2.pdf`

> **提示**：2023 年 WHO 发布了第三版指南 (mhGAP 2023)，但目前 V2.0 仍是使用最广泛的版本。
> 第三版下载页：https://www.who.int/publications/i/item/9789240084278

---

### 1.3 OpenStax Psychology 2e（开源心理学教材）

| 项目 | 内容 |
|------|------|
| **全称** | OpenStax Psychology 2e (第二版) |
| **语言** | 英文 |
| **格式** | PDF, 约 900 页 |
| **许可** | CC BY 4.0，完全免费开源 |

#### 下载步骤

```bash
mkdir -p datasets/openstax_psychology

# 从 OpenStax 官方下载（约 50MB）
wget -O datasets/openstax_psychology/Psychology2e-WEB.pdf \
  "https://assets.openstax.org/oscms-prodcms/media/documents/Psychology2e-WEB.pdf"
```

如果需要手动下载：
1. 访问：https://openstax.org/details/books/psychology-2e
2. 点击 "Get this book" → 选择 "PDF" 下载
3. 保存到 `datasets/openstax_psychology/Psychology2e-WEB.pdf`

> **建议**：这本书有 16 章，对于本项目最相关的章节是：
> - Ch.15 Psychological Disorders（心理障碍）
> - Ch.16 Therapy and Treatment（治疗）
> - Ch.10 Emotion and Motivation（情绪与动机）
> - Ch.14 Stress, Lifestyle, and Health（压力与健康）
> 
> 你可以在预处理时只提取这几章的内容以减少噪声。

---

## 二、优先级二：共情策略库

### 2.1 CPsyCoun 数据集（CAS-SIAT-XinHai 团队）

| 项目 | 内容 |
|------|------|
| **论文** | *CPsyCoun: A Report-based Multi-turn Dialogue Reconstruction and Evaluation Framework for Chinese Psychological Counseling* (ACL 2024 Findings) |
| **规模** | CPsyCounD: 3,134 条高质量多轮咨询对话 |
| **格式** | JSON (LLaMA-Factory 格式) |
| **许可** | Apache-2.0 |

#### 获取方式（二选一）

**方式 A：从 HuggingFace 下载 CPsyCounD（推荐）**

```bash
mkdir -p datasets/cpsycoun

# 使用 datasets 库下载
python -c "
from datasets import load_dataset
ds = load_dataset('CAS-SIAT-XinHai/CPsyCoun')
# 保存为 JSON
ds['train'].to_json('datasets/cpsycoun/CPsyCounD.json', force_ascii=False)
print(f'下载完成: {len(ds[\"train\"])} 条多轮对话')
"
```

或者用 `huggingface-cli`：

```bash
huggingface-cli download CAS-SIAT-XinHai/CPsyCoun \
  --repo-type dataset --local-dir datasets/cpsycoun
```

**方式 B：从 GitHub 仓库获取**

```bash
git clone https://github.com/CAS-SIAT-XinHai/CPsyCoun.git /tmp/CPsyCoun
cp /tmp/CPsyCoun/CPsyCounD/*.json datasets/cpsycoun/
```

#### 数据格式示例

CPsyCounD 为 LLaMA-Factory 格式的多轮对话：

```json
{
  "conversations": [
    {"role": "client", "content": "最近工作压力很大，经常失眠..."},
    {"role": "counselor", "content": "听起来你最近承受了不少压力，失眠一定让你感到很疲惫..."},
    {"role": "client", "content": "是的，而且白天注意力也很难集中..."},
    {"role": "counselor", "content": "持续的睡眠不足确实会影响到白天的状态..."}
  ]
}
```

> **在本项目中的用途**：我们不将 CPsyCoun 用于向量检索，而是从中提取咨询师（counselor）的
> 优质回复作为 Few-shot 共情示例，注入 Prompt 中指导模型的回应风格。

---

## 三、优先级三：安全护栏库

### 3.1 PHQ-9 / GAD-7 量表科普说明

| 项目 | 内容 |
|------|------|
| **性质** | 自行编写的科普说明文档（非量表原文！） |
| **重要声明** | 我们**不使用量表原文和计分规则**，仅提供科普级别的介绍 |

#### 为什么不直接使用量表原文？

PHQ-9 由 Pfizer 公司开发（虽然免费使用但有使用规范），GAD-7 同理。在 RAG 系统中直接嵌入量表题目和计分标准有以下风险：
- 用户可能将系统当作「自动诊断工具」使用
- 脱离专业临床场景的自评结果容易被误解
- 不符合本系统「不涉及临床诊断」的定位

因此我们只提供**科普级别的说明**，介绍量表是什么、用途、局限性，并强制附加免责声明。

#### 创建方法

这些文件需要你自行编写，下面是参考模板：

```bash
mkdir -p datasets/guardrails
```

**datasets/guardrails/phq9_intro.txt**

```
PHQ-9 科普说明

PHQ-9（患者健康问卷-9）是一种被广泛使用的抑郁症状自我筛查工具。它由 9 个问题
组成，涵盖情绪、兴趣、睡眠、精力、食欲、自我评价、注意力、行动力和自伤想法
等维度。PHQ-9 最初由 Drs. Robert L. Spitzer, Janet B.W. Williams, Kurt Kroenke
及同事于 1999 年开发，目的是为初级医疗保健场景提供简便的抑郁症状筛查。

PHQ-9 的主要用途包括：
- 在初级医疗保健场景中初步筛查抑郁症状
- 帮助医疗专业人员了解患者近两周的情绪状态
- 作为研究工具评估群体抑郁症状的严重程度

需要特别强调的局限性：
- PHQ-9 的结果不能替代专业诊断
- 任何筛查结果都应由具有资质的精神科医生或临床心理师进行综合评估
- 不建议个人在没有专业指导的情况下仅凭分数做出判断
- 量表有文化适应性的局限，不同文化背景下可能存在差异
```

**datasets/guardrails/gad7_intro.txt**

```
GAD-7 科普说明

GAD-7（广泛性焦虑障碍量表-7）是一种焦虑症状的简短自我筛查工具，包含 7 个问题，
评估个体在过去两周内焦虑症状的频率和严重程度。该量表由 Spitzer, Kroenke,
Williams 和 Löwe 于 2006 年开发。

GAD-7 的主要用途包括：
- 在社区健康筛查中初步识别可能需要进一步评估的个体
- 在临床研究中评估焦虑症状的严重程度
- 辅助医疗专业人员进行初步的焦虑症状评估

需要特别强调的局限性：
- GAD-7 仅为筛查工具，不能作为确诊依据
- 具体诊断需要由专业人员通过临床访谈和综合评估来完成
- 焦虑症状可能与多种心理和躯体疾病重叠，需要专业鉴别
- 该量表主要针对广泛性焦虑，对其他类型焦虑障碍的敏感性有限
```

---

## 四、一键下载脚本

项目中提供了自动化下载脚本 `scripts/download_datasets.py`，可一键下载所有可自动获取的数据集：

```bash
# 安装下载依赖
pip install datasets huggingface_hub requests tqdm

# 运行下载脚本
python scripts/download_datasets.py
```

脚本会：
1. ✅ 自动下载 PsyQA（HuggingFace 镜像）
2. ✅ 自动下载 CPsyCounD（HuggingFace）
3. ✅ 自动下载 WHO mhGAP-IG PDF
4. ✅ 自动下载 OpenStax Psychology PDF
5. ⚠️ 安全护栏文档需手动创建（脚本会生成模板）

---

## 五、预处理脚本

下载完成后，运行预处理脚本将原始数据转换为系统可用的格式：

```bash
python scripts/preprocess_datasets.py
```

预处理脚本做的事情：
1. PsyQA → 提取 question + answer，切块后带 metadata 存储
2. mhGAP-IG PDF → 提取文本内容，按章节切块
3. OpenStax PDF → 提取相关章节文本，切块
4. CPsyCounD → 提取 counselor 回复中的优质共情模板
5. 安全护栏 → 带 `requires_disclaimer: True` 的强 metadata 标记

---

## 六、数据集引用

如果你在论文中使用了这些数据集，请按照各数据集的要求进行引用：

```bibtex
% PsyQA
@inproceedings{sun2021psyqa,
  title={PsyQA: A Chinese Dataset for Generating Long Counseling Text 
         for Mental Health Support},
  author={Sun, Hao and Lin, Zhenru and Zheng, Chujie and Liu, Siyang 
          and Huang, Minlie},
  booktitle={Findings of the Association for Computational Linguistics: 
             ACL-IJCNLP 2021},
  pages={1489--1503},
  year={2021}
}

% CPsyCoun
@inproceedings{zhang-etal-2024-cpsycoun,
  title={{CP}sy{C}oun: A Report-based Multi-turn Dialogue Reconstruction 
         and Evaluation Framework for {C}hinese Psychological Counseling},
  author={Zhang, Chenhao and Li, Renhao and Tan, Minghuan and Yang, Min 
          and Zhu, Jingwei and Yang, Di and Zhao, Jiahao and Ye, Guancheng 
          and Li, Chengming and Hu, Xiping},
  booktitle={Findings of the Association for Computational Linguistics: 
             ACL 2024},
  pages={13947--13966},
  year={2024}
}

% WHO mhGAP-IG
@misc{who2016mhgap,
  title={mhGAP Intervention Guide for Mental, Neurological and 
         Substance Use Disorders in Non-Specialized Health Settings, 
         Version 2.0},
  author={{World Health Organization}},
  year={2016},
  publisher={WHO},
  url={https://www.who.int/publications/i/item/9789241549790}
}

% OpenStax Psychology
@book{spielman2020psychology,
  title={Psychology 2e},
  author={Spielman, Rose M and Jenkins, William J and Lovett, Marilyn D},
  year={2020},
  publisher={OpenStax, Rice University},
  url={https://openstax.org/details/books/psychology-2e}
}
```
