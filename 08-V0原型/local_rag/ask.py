"""
命令行问答入口 —— 手动体验 V0 效果（面试 Demo 也用这个）。

用法：
    python ask.py                            # 交互式问答
    python ask.py -q "请假两天找谁审批"      # 单次提问
    python ask.py -q "..." --show-chunks     # 打印检索到的原始片段（调优用）
"""

from __future__ import annotations

import argparse
import sys

from rag_core import ABSTAIN_TEXT, Config, KnowledgeBase


def main() -> int:
    ap = argparse.ArgumentParser(description="企业知识库问答 Agent（V0）")
    ap.add_argument("-q", "--question", help="单次提问（不传则进入交互模式）")
    ap.add_argument("--top-k", type=int, default=None, help="检索片段数量")
    ap.add_argument("--answer-k", type=int, default=None, help="抽取式回答拼接前 N 个片段")
    ap.add_argument("--show-chunks", action="store_true", help="展示检索片段")
    args = ap.parse_args()

    cfg = Config()
    kb = KnowledgeBase(cfg).load()
    mode = "LLM 生成" if cfg.llm_enabled else "抽取式基线（未配置 LLM_API_KEY）"
    retrieval = "混合检索（向量+关键词）" if cfg.vector_enabled else "关键词检索（BM25）"
    print(f"知识库：{len(kb.chunks)} 个片段 | 检索：{retrieval} | 生成：{mode}")
    print(f"拒答阈值 MIN_SCORE={cfg.min_score}，Top-K={args.top_k or cfg.top_k}\n")

    def handle(q: str) -> None:
        r = kb.answer(q, args.top_k, args.answer_k)
        flag = "  ⚠️ 已拒答/需人工确认" if r.need_human else ""
        print(f"【答案】{r.answer}{flag}")
        raw = r.sources[0].raw if r.sources else 0.0
        print(f"【置信度】{r.confidence}（检索原始分 {raw}，阈值 {cfg.min_score}）"
              f"   【模式】{r.mode}   【耗时】{r.latency_ms}ms", end="")
        if r.tokens_in or r.tokens_out:
            print(f"   【Token】入 {r.tokens_in} / 出 {r.tokens_out}")
        else:
            print()
        if r.sources:
            print("【来源】")
            for s in r.sources:
                print(f"  - {s.label()}  (score={s.score})")
        if args.show_chunks:
            print("【检索片段】")
            for s in r.sources:
                print(f"  ── {s.chunk_id}  (raw={s.raw}) ──\n{s.text}\n")
        print("-" * 60)

    if args.question:
        handle(args.question)
        return 0

    print("输入问题后回车（输入 q 退出）")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q:
            handle(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
