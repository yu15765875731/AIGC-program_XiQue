"""
拒答阈值标定工具 —— 决定「什么时候该说不知道」。

这是 RAG 产品最容易被忽略、却最影响企业信任度的一个参数：
    阈值太高 → 该答的拒答了（用户觉得没用）
    阈值太低 → 不该答的硬答了（用户觉得在编）

做法：对评测集里「可答题」和「不可答题」的检索分数做分布对比，
     在两者之间找一个分离点作为 MIN_SCORE。

用法：
    python calibrate.py --set ../../07-评测体系/eval_set_demo.csv
    python calibrate.py --set ../../07-评测体系/eval_set_demo.csv --plot   # 打印分数直方图
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from rag_core import Config, KnowledgeBase


def histogram(values: list[float], lo: float, hi: float, bins: int = 12, width: int = 40) -> str:
    if hi <= lo:
        return ""
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int((v - lo) / step)))
        counts[idx] += 1
    top = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(c / top * width)
        lines.append(f"  {lo + i * step:>7.2f} │{bar:<{width}} {c}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="标定拒答阈值 MIN_SCORE")
    ap.add_argument("--set", required=True, help="评测集 CSV 路径")
    ap.add_argument("--plot", action="store_true", help="打印分数直方图")
    ap.add_argument("--weight", type=float, default=3.0,
                    help="误答相对误拒的代价倍数（默认 3：宁可多拒答，不可多编造）")
    args = ap.parse_args()

    cfg = Config()
    kb = KnowledgeBase(cfg).load()
    rows = list(csv.DictReader(Path(args.set).read_text(encoding="utf-8").splitlines()))

    ans: list[tuple[float, str, str]] = []
    un: list[tuple[float, str, str]] = []
    for r in rows:
        src = kb.retrieve(r["question"])
        raw = src[0].raw if src else 0.0
        (ans if r.get("answerable", "yes") == "yes" else un).append(
            (raw, r.get("id", ""), r["question"])
        )

    def describe(name: str, data: list[tuple[float, str, str]]) -> None:
        v = sorted(x[0] for x in data)
        if not v:
            print(f"{name}: 无数据")
            return
        print(
            f"{name}（n={len(v)}）  min={v[0]:.2f}  p25={v[len(v) // 4]:.2f}  "
            f"中位数={statistics.median(v):.2f}  p75={v[3 * len(v) // 4]:.2f}  max={v[-1]:.2f}"
        )

    print(f"评测集：{Path(args.set).name}　知识库 {len(kb.chunks)} 片段\n")
    describe("可答题  ", ans)
    describe("不可答题", un)

    if not ans or not un:
        print("\n评测集中需要同时包含可答题与不可答题，才能标定阈值。")
        return 0

    a_vals = sorted(x[0] for x in ans)
    u_vals = sorted(x[0] for x in un)
    all_vals = a_vals + u_vals
    lo, hi = min(all_vals), max(all_vals)

    if args.plot:
        print("\n可答题分数分布：")
        print(histogram(a_vals, lo, hi))
        print("\n不可答题分数分布：")
        print(histogram(u_vals, lo, hi))

    # 扫描候选阈值，最小化「误拒 + 权重 × 误答」的加权误差
    print(f"\n阈值扫描（误答代价 = 误拒的 {args.weight:g} 倍）：")
    print(f"  {'阈值':>8} {'误拒(可答被判拒)':>18} {'误答(不可答被判答)':>20} {'加权误差':>10}")
    candidates = []
    for step in range(0, 81):
        th = lo + (hi - lo) * step / 80
        fr = sum(1 for v in a_vals if v < th) / len(a_vals)
        fa = sum(1 for v in u_vals if v >= th) / len(u_vals)
        candidates.append((th, fr + args.weight * fa, fr, fa))
        if step % 8 == 0:
            print(f"  {th:>8.2f} {fr:>17.1%} {fa:>19.1%} {fr + args.weight * fa:>10.2f}")

    th, score, fr, fa = min(candidates, key=lambda c: c[1])
    print(f"\n建议 MIN_SCORE = {th:.2f}")
    print(f"  该阈值下：误拒 {fr:.1%}，误答 {fa:.1%}，加权误差 {score:.2f}")
    print("\n把它写进 .env：")
    print(f"  MIN_SCORE={th:.2f}")
    print("\n注意：这是在这个小语料上的过拟合点。语料换成你自己的文档后必须重新标定。")
    print("     调高阈值 → 更保守（少编造、多拒答）；调低 → 更激进。按业务风险选，没有标准答案。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
