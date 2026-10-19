"""
企业知识库智能问答 Agent —— V0 最小可运行内核

设计原则
--------
1. 零外部依赖即可运行：BM25 关键词检索 + 抽取式回答（默认 V0 基线）
2. 配置 LLM_API_KEY 后自动升级：向量检索（混合检索）+ LLM 生成
3. 「检索 / 生成 / 兜底」三个环节可被独立评测 —— 这是 PM 需要的可观测性
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ---------------------------------------------------------------- 路径与配置

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
INDEX_PATH = BASE_DIR / "out" / "index.json"

# 兜底话术：企业级产品的底线是「不知道就说不知道」
ABSTAIN_TEXT = "未在知识库中找到可靠依据，建议联系相关职能部门确认。"

# ---------------------------------------------------------------- 环境配置


def load_env(path: Path) -> None:
    """极简 .env 解析（避免引入 python-dotenv 依赖）。已存在的环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env(BASE_DIR / ".env")


@dataclass
class Config:
    """所有可调参数集中在这里 —— 每一项都对应一次 A/B 实验。"""

    base_url: str = os.getenv("LLM_BASE_URL", "").rstrip("/")
    api_key: str = os.getenv("LLM_API_KEY", "")
    chat_model: str = os.getenv("LLM_CHAT_MODEL", "deepseek-chat")
    embed_model: str = os.getenv("LLM_EMBED_MODEL", "")
    top_k: int = int(os.getenv("TOP_K", "5"))
    # 抽取式回答时拼接前 N 个片段（模拟「把 Top-K 上下文都交给模型」）
    answer_k: int = int(os.getenv("ANSWER_K", "1"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    # 检索引擎权重（混合检索）：向量 0.6 / 关键词 0.4
    weight_vector: float = float(os.getenv("WEIGHT_VECTOR", "0.6"))
    weight_keyword: float = float(os.getenv("WEIGHT_KEYWORD", "0.4"))
    # 拒答阈值：低于该分数视为「检索不到依据」
    min_score: float = float(os.getenv("MIN_SCORE", "1.0"))
    # 置信度低于该值则标记需人工确认
    human_threshold: float = float(os.getenv("HUMAN_THRESHOLD", "0.6"))
    # 置信度换算的参考分：原始检索分达到该值时置信度记为 1.0（按语料分布标定）
    conf_ref: float = float(os.getenv("CONF_REF", "24"))
    timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key and self.base_url)

    @property
    def vector_enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.embed_model)


# ---------------------------------------------------------------- 中文分词

try:  # jieba 为可选依赖，缺失时自动降级为「汉字二元组」切分
    import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """分词策略由 RAG_TOKENIZER 控制：auto（默认，有 jieba 就用）/ jieba / bigram。

    这是留给你的第一个 A/B 实验开关：不同分词对检索排序的影响并不直观，要用评测集说话。
    """
    mode = os.getenv("RAG_TOKENIZER", "auto").lower()
    use_jieba = _HAS_JIEBA and mode in ("auto", "jieba")

    if use_jieba:
        out = []
        for w in jieba.cut(text.lower()):
            w = w.strip()
            if not w:
                continue
            if _CJK_RE.search(w) or _LATIN_RE.fullmatch(w):
                out.append(w)
        return out

    # 降级实现：拉丁词 + 汉字二元组（对 BM25 足够好，且零依赖）
    tokens: list[str] = []
    tokens.extend(m.group(0).lower() for m in _LATIN_RE.finditer(text))
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return tokens


# ---------------------------------------------------------------- 文档切片


def _split_blocks(text: str) -> list[dict]:
    """按 Markdown 标题与空行切块，并记录每块的「标题路径」（保留语义上下文）。"""
    heading_path: list[str] = []
    blocks: list[dict] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            blocks.append({"heading": " > ".join(heading_path), "text": "\n".join(buf).strip()})
            buf.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            heading_path = heading_path[: level - 1] + [title]
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush()
    return [b for b in blocks if b["text"]]


def _split_long(text: str, limit: int) -> list[str]:
    """超长块按行再切，避免单个片段超出模型上下文。"""
    parts, buf, size = [], [], 0
    for line in text.splitlines():
        if size + len(line) > limit and buf:
            parts.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        parts.append("\n".join(buf))
    return parts


def build_chunks(text: str, source: str, chunk_size: int, overlap: int) -> list[dict]:
    """切片策略：标题/段落优先 → 超长再按字数切 → 相邻片段保留 overlap 重叠。"""
    pieces: list[dict] = []
    for b in _split_blocks(text):
        if len(b["text"]) <= chunk_size:
            pieces.append(b)
        else:
            for part in _split_long(b["text"], chunk_size):
                pieces.append({"heading": b["heading"], "text": part})

    chunks: list[dict] = []
    i, n = 0, len(pieces)
    while i < n:
        heading, buf, size, j = pieces[i]["heading"], [], 0, i
        while j < n and (size + len(pieces[j]["text"]) <= chunk_size or not buf):
            buf.append(pieces[j]["text"])
            size += len(pieces[j]["text"])
            j += 1
        body = "\n".join(buf)
        chunks.append(
            {
                "source": source,
                "heading": heading,
                "text": body,
                "chars": len(body),
            }
        )
        if j >= n:
            break
        # 计算重叠：从尾部回退若干块，累计到 overlap 字符
        back, acc, k = 0, 0, j - 1
        while k > i and acc < overlap:
            acc += len(pieces[k]["text"])
            back += 1
            k -= 1
        i = max(i + 1, j - back)

    for idx, c in enumerate(chunks):
        c["id"] = f"{source}#{idx}"
    return chunks


# ---------------------------------------------------------------- 检索引擎


class BM25:
    """标准 BM25 —— 关键词检索，擅长编号、数字、专有名词的精确匹配。"""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.n = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.n)
        self.tf: list[dict] = []
        df: dict[str, int] = {}
        for d in docs_tokens:
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.n - v + 0.5) / (v + 0.5)) for t, v in df.items()}

    def score(self, q_tokens: list[str], i: int) -> float:
        dl = len(self.docs[i]) or 1
        s = 0.0
        for t in q_tokens:
            f = self.tf[i].get(t)
            if not f:
                continue
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def search(self, q_tokens: list[str], top_k: int) -> list[tuple[float, int]]:
        ranked = sorted(((self.score(q_tokens, i), i) for i in range(self.n)), reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------- LLM 客户端


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LLMClient:
    """OpenAI 兼容接口客户端（DeepSeek / 通义 / OpenAI / 本地 vLLM 均可）。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def chat(self, system: str, user: str) -> tuple[str, dict]:
        payload = {
            "model": self.cfg.chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,  # 知识问答是「抽取/判断」类任务，温度要低
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }
        try:
            data = _post_json(
                f"{self.cfg.base_url}/chat/completions",
                payload,
                {"Authorization": f"Bearer {self.cfg.api_key}"},
                self.cfg.timeout,
            )
        except urllib.error.HTTPError as e:  # response_format 不被支持时降级重试
            if e.code in (400, 422):
                payload.pop("response_format", None)
                data = _post_json(
                    f"{self.cfg.base_url}/chat/completions",
                    payload,
                    {"Authorization": f"Bearer {self.cfg.api_key}"},
                    self.cfg.timeout,
                )
            else:
                raise
        usage = data.get("usage", {}) or {}
        return data["choices"][0]["message"]["content"], {
            "in": usage.get("prompt_tokens", 0),
            "out": usage.get("completion_tokens", 0),
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post_json(
            f"{self.cfg.base_url}/embeddings",
            {"model": self.cfg.embed_model, "input": texts},
            {"Authorization": f"Bearer {self.cfg.api_key}"},
            self.cfg.timeout,
        )
        return [item["embedding"] for item in data["data"]]


# ---------------------------------------------------------------- 知识库


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class Source:
    doc: str
    chunk_id: str
    heading: str
    snippet: str
    score: float
    raw: float = 0.0  # 原始检索分数（未归一化），用于拒答阈值判断
    text: str = ""    # 完整片段正文（生成与评测用）

    def label(self) -> str:
        return f"{self.doc} > {self.heading}" if self.heading else self.doc


@dataclass
class Result:
    question: str
    answer: str
    confidence: float
    sources: list[Source] = field(default_factory=list)
    need_human: bool = False
    abstained: bool = False
    mode: str = "extractive"
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_labels"] = [s.label() for s in self.sources]
        return d


SYSTEM_PROMPT = """你是企业知识库问答助手。
你的回答必须严格基于用户提供的【知识片段】，不得编造任何事实。
如果【知识片段】不足以回答，你必须回答"未在知识库中找到可靠依据"，不得猜测。"""

USER_TEMPLATE = """【知识片段】
{context}

【用户问题】
{question}

【输出要求】
仅输出一个 JSON 对象，不要任何多余文字，格式如下：
{{"answer": "答案正文（简洁、分点、可执行）", "confidence": 0.95, "sources": ["引用片段的来源编号，如 1"], "need_human": false}}
说明：
- confidence 为 0 到 1 的小数，表示你对答案的把握
- 若知识片段不足以回答，answer 固定为"未在知识库中找到可靠依据"，confidence 设为 0，sources 为空数组
- 绝不使用【知识片段】之外的信息"""


class KnowledgeBase:
    def __init__(self, cfg: Config | None = None, index_path: Path | None = None):
        self.cfg = cfg or Config()
        self.index_path = index_path or INDEX_PATH
        self.chunks: list[dict] = []
        self.bm25: BM25 | None = None
        self.vectors: list[list[float]] = []
        self.llm = LLMClient(self.cfg) if self.cfg.llm_enabled else None

    # -------------------------------------------------- 索引

    def load(self) -> "KnowledgeBase":
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"索引不存在：{self.index_path}\n请先运行：python ingest.py"
            )
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.chunks = payload["chunks"]
        self.vectors = payload.get("vectors", [])
        self.bm25 = BM25([tokenize(self._searchable(c)) for c in self.chunks])
        return self

    @staticmethod
    def _searchable(chunk: dict) -> str:
        # 标题路径拼进正文，让「第二章 请假与考勤」这类上下文可被检索到
        return f"{chunk.get('heading', '')} {chunk['text']}"

    # -------------------------------------------------- 检索

    def retrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        top_k = top_k or self.cfg.top_k
        assert self.bm25 is not None, "请先 load()"

        kw = self.bm25.search(tokenize(question), max(top_k * 4, 20))
        # 归一化关键词分数到 0-1（仅用于排序与展示）
        max_kw = max((s for s, _ in kw), default=0.0) or 1.0
        kw_map = {i: s / max_kw for s, i in kw}
        raw_map = {i: s for s, i in kw}

        vec: dict[int, float] = {}
        if self.vectors and self.llm and self.cfg.vector_enabled:
            try:
                qv = self.llm.embed([question])[0]
                sims = [(_cosine(qv, v), i) for i, v in enumerate(self.vectors)]
                sims.sort(reverse=True)
                max_v = max((s for s, _ in sims[: top_k * 4]), default=1.0) or 1.0
                vec = {i: s / max_v for s, i in sims[: top_k * 4]}
            except Exception:
                vec = {}  # 向量服务异常 → 静默降级为纯关键词检索

        hybrid = bool(vec)
        pool = set(kw_map) | set(vec) if hybrid else set(kw_map)
        scored: list[tuple[float, int]] = []
        for i in pool:
            k = kw_map.get(i, 0.0)
            v = vec.get(i, 0.0)
            score = (
                self.cfg.weight_vector * v + self.cfg.weight_keyword * k if hybrid else k
            )
            scored.append((score, i))
        scored.sort(reverse=True)

        return [
            Source(
                doc=self.chunks[i]["source"],
                chunk_id=self.chunks[i]["id"],
                heading=self.chunks[i]["heading"],
                snippet=self.chunks[i]["text"][:200],
                score=round(s, 4),
                raw=round(raw_map.get(i, 0.0), 4),
                text=self.chunks[i]["text"],
            )
            for s, i in scored[:top_k]
        ]

    # -------------------------------------------------- 问答主链路

    def answer(self, question: str, top_k: int | None = None, answer_k: int | None = None) -> Result:
        t0 = time.perf_counter()
        question = (question or "").strip()
        k_ans = max(1, answer_k or self.cfg.answer_k)
        if not question:
            return Result(question=question, answer="请输入问题。", confidence=0.0, abstained=True)

        sources = self.retrieve(question, top_k)
        # 用「原始（未归一化）检索分数」判拒答与置信度：
        # 归一化后 Top1 恒为 1.0，会失去区分度
        top_raw = sources[0].raw if sources else 0.0
        conf = self.confidence_of(top_raw)

        # ⑤ 兜底：检索不到依据时不调 LLM —— 宁可不答，不可编造
        if not sources or top_raw < self.cfg.min_score:
            return Result(
                question=question,
                answer=ABSTAIN_TEXT,
                confidence=conf,
                sources=sources,
                need_human=True,
                abstained=True,
                mode="abstain",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )

        base = dict(sources=sources, latency_ms=0, confidence=conf)

        if not self.llm:
            # V0 抽取式基线：直接返回最相关片段的原文（可独立评测「检索」质量）
            # answer_k > 1 时拼接前 N 个片段，模拟「把 Top-K 上下文交给模型」
            body = "\n\n---\n\n".join(s.text for s in sources[:k_ans])
            res = Result(
                question=question,
                answer=body,
                need_human=conf < self.cfg.human_threshold,
                mode=f"extractive_top{k_ans}",
                **base,
            )
            res.latency_ms = int((time.perf_counter() - t0) * 1000)
            return res

        # ⑦ 上下文组装：编号 + 标题路径 + 正文（单片段截断以控制 Token）
        context = "\n\n".join(
            f"[{i + 1}] 来源：{s.label()}\n{s.text[:600]}" for i, s in enumerate(sources)
        )
        try:
            raw, usage = self.llm.chat(
                SYSTEM_PROMPT, USER_TEMPLATE.format(context=context, question=question)
            )
            data = self._parse_json(raw)
            text = str(data.get("answer", "")).strip() or ABSTAIN_TEXT
            conf = float(data.get("confidence", conf) or 0.0)
            abstained = "未找到可靠依据" in text or not data.get("sources")
            res = Result(
                question=question,
                answer=text,
                confidence=round(conf, 4),
                need_human=bool(data.get("need_human")) or conf < self.cfg.human_threshold,
                abstained=abstained,
                mode="llm",
                tokens_in=usage.get("in", 0),
                tokens_out=usage.get("out", 0),
                **base,
            )
        except Exception as e:  # ⑨ 降级：LLM 失败 → 退回抽取式，绝不空手而归
            res = Result(
                question=question,
                answer="\n\n---\n\n".join(s.text for s in sources[:k_ans]),
                need_human=True,
                mode="extractive_fallback",
                error=f"{type(e).__name__}: {e}",
                **base,
            )
        res.latency_ms = int((time.perf_counter() - t0) * 1000)
        return res

    def confidence_of(self, raw_top: float) -> float:
        """把原始检索分数换算为 0~1 的置信度。

        参考分 conf_ref 由语料分布标定（默认为可答题分数的 p75 左右）。
        这样 confidence 才有区分度 —— 直接用归一化分数的话 Top1 恒为 1.0。
        """
        return round(min(1.0, raw_top / max(self.cfg.conf_ref, 1e-6)), 4)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {"answer": text, "confidence": 0.0, "sources": [], "need_human": True}


if __name__ == "__main__":  # 快速冒烟测试：python rag_core.py
    kb = KnowledgeBase().load()
    print("加载片段数：", len(kb.chunks))
    for r in kb.retrieve("请假两天找谁审批", 3):
        print(f"  {r.score:.3f}  {r.label()}")
