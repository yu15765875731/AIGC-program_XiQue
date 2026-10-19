"""
评测脚本 —— 拿一套标准题目给 RAG 系统打分。

这是整个项目里最重要的一环：没有评测，所有「优化」都只是感觉。

用法：
    python evaluate.py --set ../../07-评测体系/eval_set_demo.csv --tag baseline
    python evaluate.py --set ../../07-评测体系/eval_set_demo.csv --tag v1 --top-k 5 --judge

指标定义（详见 ../../07-评测体系/README.md）：
    检索 Hit@K      —— 期望文档是否落入 Top-K 召回（先看「找没找到」）
    要点召回率      —— 期望要点在召回片段中出现的比例（看「找得够不够」）
    答案要点覆盖率  —— 期望要点在最终答案中出现的比例（看「答得对不对」）
    答案正确率      —— 要点覆盖率 ≥ 阈值的题目占比
    拒答准确率      —— 不可答题中「正确说不知道」的比例 ⭐
    幻觉率          —— 不可答题中「硬答」的比例 ⭐ 企业最怕这个
    引用率          —— 回答附带来源的比例
    P50/P95 延迟    —— 体验指标
    单次成本        —— 用真实 Token 数按单价折算
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from rag_core import ABSTAIN_TEXT, Config, KnowledgeBase, Result

# 模型单价（元 / 百万 Token），按你实际使用的模型改
PRICE_IN = 2.0
PRICE_OUT = 8.0

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize(text: str) -> str:
    """比对前的文本归一化。

    这一步看着不起眼，但极其关键：文档里常写「提前 1 天」，评测要点写「提前1天」，
    如果不去空格就会「明明答对了却判错」——那你测的是排版格式，不是产品能力。
    """
    return re.sub(r"\s+", "", (text or "").translate(_FULLWIDTH_DIGITS)).lower()


def parse_groups(raw: str) -> list[list[str]]:
    """把 `要点A/同义表达|要点B` 解析为 [[要点A, 同义表达], [要点B]]，任一命中即算该组命中。"""
    groups: list[list[str]] = []
    for part in (raw or "").split("|"):
        alts = [a.strip() for a in part.split("/") if a.strip()]
        if alts:
            groups.append(alts)
    return groups


def coverage(groups: list[list[str]], text: str) -> float:
    if not groups:
        return 1.0
    haystack = normalize(text)
    hit = sum(1 for alts in groups if any(normalize(a) in haystack for a in alts))
    return hit / len(groups)


def is_abstain(r: Result) -> bool:
    return r.abstained or ABSTAIN_TEXT[:8] in r.answer or "未找到" in r.answer


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100 * len(ordered))) - 1))
    return ordered[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 评测")
    ap.add_argument("--set", required=True, help="评测集 CSV 路径")
    ap.add_argument("--tag", default="run", help="本次运行的标签（用于文件名与对比）")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--answer-k", type=int, default=None,
                    help="抽取式回答拼接前 N 个片段（无 LLM 时做上下文对比实验）")
    ap.add_argument("--correct-threshold", type=float, default=0.6, help="要点覆盖率达到该值判为答对")
    ap.add_argument("--judge", action="store_true", help="额外调用 LLM 作为裁判打分（需 API Key）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    args = ap.parse_args()

    cfg = Config()
    kb = KnowledgeBase(cfg).load()
    rows = list(csv.DictReader(Path(args.set).read_text(encoding="utf-8").splitlines()))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("评测集为空")
        return 1

    print(f"评测集：{Path(args.set).name}  共 {len(rows)} 条")
    print(f"检索：{'混合（向量+关键词）' if cfg.vector_enabled else '关键词 BM25'}  "
          f"生成：{'LLM' if cfg.llm_enabled else '抽取式基线'}  Top-K={args.top_k or cfg.top_k}")
    print("-" * 78)

    records: list[dict] = []
    t_start = time.perf_counter()

    for n, row in enumerate(rows, 1):
        q = row["question"].strip()
        answerable = row.get("answerable", "yes").strip().lower() in ("yes", "y", "1", "是")
        exp_sources = [s.strip() for s in (row.get("expected_sources") or "").split("|") if s.strip()]
        groups = parse_groups(row.get("expected_keywords", ""))

        r = kb.answer(q, args.top_k, args.answer_k)
        snippet_text = "\n".join(s.text for s in r.sources)
        src_docs = [s.doc for s in r.sources]

        rec = {
            "id": row.get("id", f"R{n}"),
            "category": row.get("category", ""),
            "question": q,
            "answerable": answerable,
            "expected_answer": row.get("expected_answer", ""),
            "answer": r.answer.replace("\n", " "),
            "abstained": is_abstain(r),
            # 来源匹配用「包含」而非全等：文档名可能带 01- 之类的序号前缀
            "hit_at_k": (not exp_sources)
            or any(normalize(e) in normalize(d) or normalize(d) in normalize(e)
                   for d in src_docs for e in exp_sources),
            "kw_recall_retrieval": round(coverage(groups, snippet_text), 4),
            "kw_coverage_answer": round(coverage(groups, r.answer), 4),
            "citation": bool(r.sources),
            "confidence": r.confidence,
            "need_human": r.need_human,
            "latency_ms": r.latency_ms,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "mode": r.mode,
            "top_sources": " | ".join(r.sources[:3][i].label() for i in range(min(3, len(r.sources)))),
            "error": r.error,
        }
        rec["correct"] = (
            rec["kw_coverage_answer"] >= args.correct_threshold if answerable else rec["abstained"]
        )
        rec["hallucination"] = (not answerable) and (not rec["abstained"])
        if args.judge:
            rec["judge_score"] = _judge(kb, row, r)
        records.append(rec)

        mark = "OK " if rec["correct"] else "BAD"
        print(f"[{n:>3}/{len(rows)}] {mark} {rec['id']:<6}{rec['category']:<13}"
              f"hit={int(rec['hit_at_k'])} cov={rec['kw_coverage_answer']:.2f} "
              f"{rec['latency_ms']:>5}ms  {q[:26]}")

    stats = _summarize(records, args.correct_threshold)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"results_{args.tag}.csv").write_text(
        _to_csv(records), encoding="utf-8-sig"
    )
    report = _report(args.tag, Path(args.set).name, cfg, len(kb.chunks), stats, records)
    (out_dir / f"report_{args.tag}.md").write_text(report, encoding="utf-8")

    print("-" * 78)
    print(f"耗时 {time.perf_counter() - t_start:.1f}s")
    for k, v in stats["overall"].items():
        print(f"  {k}: {v}")
    print(f"\n结果：out/results_{args.tag}.csv")
    print(f"报告：out/report_{args.tag}.md")
    return 0


def _judge(kb: KnowledgeBase, row: dict, r: Result) -> str:
    """LLM-as-a-Judge：按「是否正确且仅依据知识库」打 0-2 分。"""
    if not kb.llm:
        return "n/a"
    prompt = (
        f"问题：{row['question']}\n"
        f"参考答案要点：{row.get('expected_answer', '')}\n"
        f"该问题是否可依据知识库回答：{'是' if row.get('answerable') == 'yes' else '否'}\n"
        f"系统回答：{r.answer[:600]}\n\n"
        "请只输出 JSON：{\"score\": 0|1|2, \"reason\": \"...\"}\n"
        "2=完全正确；1=部分正确或有瑕疵；0=错误、编造，或对不可答题未诚实拒答。"
    )
    try:
        raw, _ = kb.llm.chat("你是严格的评测裁判，只依据给定标准打分。", prompt)
        return str(json.loads(raw).get("score"))
    except Exception as e:
        return f"err:{type(e).__name__}"


def _summarize(records: list[dict], threshold: float) -> dict:
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]
    lat = [r["latency_ms"] for r in records]
    tin = sum(r["tokens_in"] for r in records)
    tout = sum(r["tokens_out"] for r in records)
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT

    overall = {
        "题目总数": len(records),
        "答案正确率": f"{_pct(ans, 'correct')}",
        "检索 Hit@K": f"{_pct(ans, 'hit_at_k')}",
        "要点召回率": f"{_avg(ans, 'kw_recall_retrieval'):.3f}",
        "答案要点覆盖率": f"{_avg(ans, 'kw_coverage_answer'):.3f}",
        "拒答准确率": f"{_pct(unans, 'correct')}",
        "幻觉率": f"{_pct(unans, 'hallucination')}",
        "引用率": f"{_pct(records, 'citation')}",
        "P50 延迟(ms)": int(percentile(lat, 50)),
        "P95 延迟(ms)": int(percentile(lat, 95)),
        "总 Token": f"入 {tin} / 出 {tout}",
        "平均单次成本(元)": f"{cost / len(records):.6f}" if tin or tout else "0（未调用 LLM）",
    }

    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in records}):
        sub = [r for r in records if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "正确率": f"{_pct(sub, 'correct')}",
            "Hit@K": f"{_pct([r for r in sub if r['answerable']] or sub, 'hit_at_k')}",
            "要点覆盖率": f"{_avg(sub, 'kw_coverage_answer'):.3f}",
        }
    return {"overall": overall, "by_category": by_cat}


def _pct(rows: list[dict], field: str) -> str:
    if not rows:
        return "n/a"
    return f"{sum(1 for r in rows if r[field]) / len(rows):.1%}"


def _avg(rows: list[dict], field: str) -> float:
    return statistics.fmean([r[field] for r in rows]) if rows else 0.0


def _to_csv(records: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()


def _report(tag: str, set_name: str, cfg: Config, n_chunks: int, s: dict, records: list[dict]) -> str:
    lines = [
        f"# 评测报告 · {tag}",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 评测集：`{set_name}`（{len(records)} 条）",
        f"- 知识库：{n_chunks} 个片段",
        f"- 检索：{'混合（向量 0.6 + 关键词 0.4）' if cfg.vector_enabled else '关键词 BM25'}",
        f"- 生成：{'LLM ' + cfg.chat_model if cfg.llm_enabled else '抽取式基线（返回 Top1 原文片段）'}",
        f"- Top-K：{cfg.top_k}　拒答阈值 MIN_SCORE：{cfg.min_score}"
        f"　抽取拼接片段数：{cfg.answer_k}",
        "",
        "## 一、总体指标",
        "",
        "| 指标 | 值 |",
        "|------|----|",
    ]
    lines += [f"| {k} | {v} |" for k, v in s["overall"].items()]

    lines += ["", "## 二、分类别表现", "", "| 类别 | 题量 | 正确率 | Hit@K | 要点覆盖率 |", "|------|------|--------|-------|-----------|"]
    lines += [
        f"| {c} | {v['n']} | {v['正确率']} | {v['Hit@K']} | {v['要点覆盖率']} |"
        for c, v in s["by_category"].items()
    ]

    bad = [r for r in records if not r["correct"]][:20]
    lines += ["", f"## 三、Bad Case（{len([r for r in records if not r['correct']])} 条，列出前 20）", ""]
    if not bad:
        lines.append("无。")
    for r in bad:
        lines += [
            f"### {r['id']}　{r['category']}　{r['question']}",
            f"- 是否可答：{'是' if r['answerable'] else '否'}　Hit@K：{'✅' if r['hit_at_k'] else '❌'}"
            f"　要点覆盖率：{r['kw_coverage_answer']}",
            f"- 期望答案：{r['expected_answer'][:120]}",
            f"- 实际回答：{r['answer'][:200]}",
            f"- 实际召回：{r['top_sources']}",
            "",
        ]

    lines += [
        "## 四、下一步优化方向（人工填写）",
        "",
        "> 别只看总分。对着上面的 Bad Case 逐条归因，判断是下面哪一类问题，再决定改什么。",
        "",
        "| 归因 | 判据 | 对应动作 |",
        "|------|------|---------|",
        "| 检索失败 | Hit@K ❌ | 改切片策略 / 加同义词 / 上混合检索 |",
        "| 检索够但没答对 | Hit@K ✅ 但要点覆盖率低 | 改 Prompt / 加 Few-shot / 换模型 |",
        "| 知识缺失 | 期望内容根本不在库里 | 补文档，不是改模型 |",
        "| 该拒答却硬答 | 幻觉率 > 0 | 调 MIN_SCORE / 强化 Prompt 约束 |",
        "| 该答却拒答 | 正常题 abstain | 阈值过高，需下调 |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
