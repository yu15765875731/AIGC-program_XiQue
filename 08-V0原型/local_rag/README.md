# local_rag —— 本机可跑的 RAG V0 内核

> 一个**零外部依赖就能跑**的企业知识库问答最小实现。
> 配了 API Key 自动升级为「向量检索 + LLM 生成」，没配也能跑「关键词检索 + 抽取式回答」。

---

## 为什么要有它

写 PRD 只能证明你会写文档，跑出数字才能证明你会做产品。

这个内核的设计目标是**让「检索 / 生成 / 兜底」三个环节可以被独立评测**——
因为 PM 的核心动作是「归因」：答错了，到底该改检索、改 Prompt、还是补文档？

---

## 快速开始

```powershell
cd 08-V0原型\local_rag

# 0) 虚拟环境（我已经建好了 .venv，如已存在可跳过）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1) 建索引
.\.venv\Scripts\python.exe ingest.py

# 2) 问答
.\.venv\Scripts\python.exe ask.py -q "实验室安全考试的合格线是多少分"
.\.venv\Scripts\python.exe ask.py                     # 交互模式
.\.venv\Scripts\python.exe ask.py -q "..." --show-chunks   # 看检索到了什么

# 3) 标定拒答阈值（换语料后必做）
.\.venv\Scripts\python.exe calibrate.py --set ../../07-评测体系/eval_set_demo.csv --plot

# 4) 跑评测
.\.venv\Scripts\python.exe evaluate.py --set ../../07-评测体系/eval_set_demo.csv --tag v0
```

---

## 文件结构

```
local_rag/
├── rag_core.py        # 核心：分词 → 切片 → BM25/向量检索 → Prompt → 生成 → 降级
├── ingest.py          # 建索引（支持 --chunk-size / --overlap 做切片 A/B）
├── ask.py             # 命令行问答（面试 Demo 用这个）
├── evaluate.py        # 评测：6 项指标 + 分类表现 + Bad Case 清单
├── calibrate.py       # 拒答阈值标定（误拒 vs 误答的权衡曲线）
├── requirements.txt   # 唯一可选依赖：jieba
├── .env.example       # 配置模板
├── .env               # 你的实际配置（含 Key，别提交）
├── data/docs/         # 知识库文档（换成你自己的）
└── out/               # index.json + report_*.md + results_*.csv
```

---

## 参数说明（每一个都对应一次实验）

| 参数 | 默认 | 含义 | 改了会怎样 |
|------|------|------|-----------|
| `CHUNK_SIZE` | 500 | 切片大小（字符） | 太小 → 语义被切碎；太大 → 检索不精准、浪费 Token |
| `CHUNK_OVERLAP` | 100 | 相邻片段重叠 | 太小 → 答案被切在边界上就找不到；太大 → 冗余、成本上升 |
| `TOP_K` | 5 | 送进模型的片段数 | 太少漏信息；太多稀释注意力并推高成本 |
| `ANSWER_K` | 1 | 抽取式回答拼接几个片段 | 无 LLM 时用它做上下文对比实验 |
| `MIN_SCORE` | 8.76 | **拒答阈值** | ⭐ 最关键的参数。高了该答不答，低了不该答硬答 |
| `HUMAN_THRESHOLD` | 0.6 | 低于此置信度标记「需人工确认」 | 影响转人工比例 |
| `WEIGHT_VECTOR` / `WEIGHT_KEYWORD` | 0.6 / 0.4 | 混合检索权重 | 只在配了 Embedding 时生效 |
| `RAG_TOKENIZER` | bigram | 分词器：`bigram` / `jieba` / `auto` | 实测在示例语料上 bigram 明显更好 |

### 三种检索模式（按配置自动切换）

```
未配 API Key
  └─ 关键词检索（BM25） + 抽取式回答          ← V0 基线，零成本

配了 API Key，未配 Embedding
  └─ 关键词检索（BM25） + LLM 生成            ← 能看出「生成」带来的提升

配了 API Key + Embedding
  └─ 混合检索（向量 0.6 + 关键词 0.4） + LLM 生成   ← 完整形态
```

---

## 问答主链路

```
用户提问
  → ① 检索（BM25 / 混合）
  → ② 拒答判断：原始分数 < MIN_SCORE → 直接说「未找到依据」，**不调 LLM**  ← 成本 + 安全双收益
  → ③ 上下文组装（编号 + 标题路径 + 正文，单片段截断 600 字符）
  → ④ LLM 生成（强制 JSON：answer / confidence / sources / need_human）
  → ⑤ 校验：无引用 → 视为失败；置信度 < 阈值 → 标记需人工确认
  → ⑥ 降级：LLM 超时/异常 → 回退返回检索原文（宁可不生成，不可编造）
```

**② 和 ⑥ 是「企业级」和「玩具」的分界线。**

---

## 换成你自己的文档

```powershell
# 1) 清空示例语料，放入你的文档（支持 .md / .txt）
Remove-Item data\docs\*.md
copy <你的文档目录>\* data\docs\

# 2) 重建索引
.\.venv\Scripts\python.exe ingest.py

# 3) 重新标定阈值（语料一变，分数分布全变，这一步千万别跳过）
.\.venv\Scripts\python.exe calibrate.py --set ../../07-评测体系/eval_set.csv

# 4) 按新阈值跑评测
.\.venv\Scripts\python.exe evaluate.py --set ../../07-评测体系/eval_set.csv --tag myv1
```

> 目前只支持 `.md` / `.txt`。
> 要处理 PDF/Word：`pip install pymupdf python-docx`，然后在 `ingest.py` 里加对应的解析分支。
> 或者直接用 Dify / RAGFlow —— **别在解析上耗掉你的学习时间**。

---

## 做实验的标准流程（这是你的作品集素材）

```powershell
# 单变量实验：每次只改一个参数，其余不动
$env:RAG_TOKENIZER = "bigram"
.\.venv\Scripts\python.exe evaluate.py --set <评测集> --tag exp_bigram

$env:RAG_TOKENIZER = "jieba"
.\.venv\Scripts\python.exe evaluate.py --set <评测集> --tag exp_jieba

# 对比 out\report_exp_bigram.md 和 out\report_exp_jieba.md 的「一、总体指标」
```

**每次实验记得记三件事**：改了什么、哪些指标动了、为什么。
这三行字就是面试时你最有说服力的内容。

---

## 已知限制（诚实清单，面试也是加分项）

| 限制 | 影响 | 生产上怎么解 |
|------|------|-------------|
| 只支持 .md/.txt | 无法直接吃 PDF | 加 PyMuPDF / Unstructured 解析 |
| 无 Embedding 时纯关键词 | 有词汇鸿沟：「2到3天」匹配不到「2 至 3 天」 | 混合检索 / Query 改写 |
| 无 Rerank | 粗排即终排，Top-1 精度受限 | 加 BGE-Reranker |
| 相似度缓存未实现 | 重复问题重复付费 | 加 embedding 相似度缓存 |
| 无多租户 / 无权限 | 单用户玩具 | 见 [04-技术架构](../04-技术架构/README.md) 的多租户方案 |
| 评测集与语料同源 | 指标会偏乐观 | 加一个从未看过的测试集 |
