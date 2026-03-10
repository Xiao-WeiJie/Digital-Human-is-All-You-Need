# 🌿 PsyMind — 心理健康科普 RAG 对话系统

> **学术研究 Demo** | LangChain LCEL + 通义千问 + FAISS 多路检索 | 模块化架构

⚠️ **声明**：本系统仅用于心理健康科普教育与学术研究，**不涉及任何临床诊断或治疗行为**。

---

## 项目结构

```
psymind/
├── main.py                          # 🚀 入口：交互式 CLI 对话循环
├── system.py                        # 🧠 系统主类：全局编排器
├── requirements.txt                 # 📦 Python 依赖
├── README.md                        # 📖 本文件
│
├── config/                          # ⚙️ 配置模块
│   ├── __init__.py
│   └── settings.py                  #    API、模型参数、数据库路径、安全常量
│
├── db/                              # 💾 数据库管理模块
│   ├── __init__.py
│   ├── vector_store.py              #    Chroma 向量数据库（持久化向量索引）
│   ├── session_store.py             #    SQLite 会话存储（对话历史、共情缓存）
│   ├── doc_registry.py              #    文档注册表（追踪已入库文件，增量更新）
│   ├── chroma_data/                 #    [自动生成] Chroma 持久化数据目录
│   └── psymind.db                   #    [自动生成] SQLite 数据库文件
│
├── prompts/                         # 📝 Prompt 工程模块
│   ├── __init__.py
│   └── templates.py                 #    System Prompt + 各意图 Addendum
│
├── data/                            # 📚 数据注入模块（三级知识库）
│   ├── __init__.py
│   ├── factual_docs.py              #    优先级一：科普核心库
│   ├── empathy_examples.py          #    优先级二：共情策略库
│   ├── guardrail_docs.py            #    优先级三：安全护栏库
│   └── real_data_loader.py          #    真实数据加载器（替换模拟数据）
│
├── retrieval/                       # 🔍 多路检索模块
│   ├── __init__.py
│   └── multi_way_retriever.py       #    Chroma 双索引 + SQLite 共情缓存
│
├── memory/                          # 🧩 对话记忆模块
│   ├── __init__.py
│   └── conversation_memory.py       #    SQLite 持久化长程记忆 + 自动摘要
│
├── routing/                         # 🔀 意图路由模块
│   ├── __init__.py
│   └── intent_classifier.py         #    LLM 语义分类 + 关键词硬规则
│
├── chains/                          # ⛓️ LCEL 对话链模块
│   ├── __init__.py
│   └── dialogue_chains.py           #    四条意图专用链的工厂类
│
└── scripts/                         # 🔧 工具脚本
    ├── download_datasets.py         #    一键下载全部数据集
    └── preprocess_datasets.py       #    数据预处理
```

## 系统架构

```
用户输入
   │
   ▼
┌──────────────────────┐
│  意图分类器            │  routing/
│  (LLM + 关键词兜底)   │
└──────────┬───────────┘
           │
   ┌───────┼───────────────┐
   ▼       ▼               ▼
情绪宣泄  知识问询      安全护栏
   │       │               │
   ▼       ▼               ▼
SQLite    Chroma         Chroma            db/
共情缓存  factual_kb     guardrail_kb
   │       │               │
   └───────┼───────────────┘
           ▼
┌──────────────────────────────┐
│  LCEL 对话链 + SQLite 记忆    │  chains/ + memory/
│  (持久化会话历史 + 自动摘要)   │
└──────────┬───────────────────┘
           ▼
       最终回复
```

## 数据库架构

```
┌─────────────────────────────────────────────────────┐
│                 持久化存储层 (db/)                    │
│                                                     │
│  ┌─────────────────────────┐  ┌──────────────────┐  │
│  │ Chroma (向量数据库)      │  │ SQLite (关系型)   │  │
│  │ db/chroma_data/         │  │ db/psymind.db    │  │
│  │                         │  │                  │  │
│  │ • factual_kb collection │  │ • sessions 表    │  │
│  │   (科普核心库向量索引)    │  │ • messages 表    │  │
│  │                         │  │ • empathy 表     │  │
│  │ • guardrail_kb          │  │ • doc_registry   │  │
│  │   (安全护栏库向量索引)    │  │                  │  │
│  └─────────────────────────┘  └──────────────────┘  │
│                                                     │
│  特性:                                               │
│  • 首次启动: 自动入库 → 后续启动: 毫秒级加载          │
│  • 数据更新: 增量入库，只写入新文档                    │
│  • 进程重启: 对话历史完整恢复                          │
│  • 管理命令: rebuild(重建) / sessions(查看历史)        │
└─────────────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 实现方式 | 所在模块 |
|------|---------|---------|
| 语义路由 | LLM 意图分类 + 关键词硬规则双重保障 | `routing/` |
| 多路检索 | Chroma 科普库 + Chroma 安全库 + SQLite 共情缓存 | `retrieval/` + `db/` |
| 向量持久化 | Chroma 磁盘存储，首次入库后毫秒级加载 | `db/vector_store.py` |
| 会话持久化 | SQLite 存储对话历史、摘要、用户名 | `db/session_store.py` |
| 增量入库 | 文档 hash 去重 + doc_registry 追踪已入库文件 | `db/doc_registry.py` |
| 长文本记忆 | SQLite 持久化 + 自动摘要压缩（>6轮触发） | `memory/` |
| 安全护栏 | 危险信号不漏判 + 强制免责声明 | `routing/` + `config/` |
| Prompt 工程 | 温暖共情风格 + 意图专用指令片段 | `prompts/` |

## 快速开始

### 1. 安装依赖

```bash
cd psymind
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 从 https://dashscope.console.aliyun.com/ 获取
export DASHSCOPE_API_KEY="sk-your-dashscope-api-key"
```

### 3. 运行

```bash
python main.py
```

### 4. 对话示例

```
🧑 你: 我最近总是失眠，白天也提不起精神
  [路由] 意图分类: emotional
  🌿 小暖: 连续失眠真的会让整个人都很疲惫...

🧑 你: 什么是 CBT？
  [路由] 意图分类: knowledge
  🌿 小暖: CBT 全称认知行为疗法...

🧑 你: PHQ-9 是什么？
  [路由] 意图分类: guardrail
  🌿 小暖: PHQ-9 是一种常见的抑郁症状筛查工具...⚠️【免责声明】...

🧑 你: debug
  [调试] 当前轮次: 3
  [调试] 历史消息数: 6
  [调试] 当前摘要: （无）
```

## 模块说明

### `config/settings.py` — 改配置只改这一个文件
所有可调参数集中管理：API 端点、模型名、温度、切块大小、检索 K 值、安全关键词等。

### `data/` — 替换真实数据集
当前使用模拟文档。替换时只需修改对应文件：

```python
# data/factual_docs.py 中替换为真实数据
from datasets import load_dataset
ds = load_dataset("thu-coai/PsyQA")
documents = [
    Document(
        page_content=item["answer"],
        metadata={"type": "factual", "source": "PsyQA", "topic": item["topic"]}
    )
    for item in ds["train"]
]
```

### `prompts/templates.py` — Prompt 迭代
所有 Prompt 模板集中在一个文件中，方便对比和版本管理。

### `routing/intent_classifier.py` — 调整路由逻辑
修改安全关键词、添加新意图类型、调整分类 Prompt 均在此模块完成。

## 技术栈

- **LLM**: 通义千问 qwen-plus（128k 上下文窗口）
- **Framework**: LangChain 0.3+ (LCEL)
- **Vector Store**: Chroma（持久化双 Collection 架构）
- **RDBMS**: SQLite（会话历史 + 共情缓存 + 文档注册）
- **Embedding**: text-embedding-v3
- **Memory**: SQLite 持久化 + LLM 自动摘要压缩
