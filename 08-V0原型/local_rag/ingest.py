"""
构建知识库索引：读取 data/docs 下的文档 → 切片 → （可选）向量化 → 落盘 out/index.json

用法：
    python ingest.py                 # 切片 + 保存（仅关键词检索所需）
    python ingest.py --with-vectors  # 额外调用 Embedding API 生成向量（混合检索用）
    python ingest.py --chunk-size 800 --overlap 200   # 切片参数 A/B 实验
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from rag_core import BASE_DIR, DOCS_DIR, INDEX_PATH, Config, KnowledgeBase, build_chunks

SUPPORTED = {".md", ".txt", ".markdown"}


def main() -> int:
    ap = argparse.ArgumentParser(description="构建本地 RAG 索引")
    ap.add_argument("--docs", default=str(DOCS_DIR), help="文档目录")
    ap.add_argument("--out", default=str(INDEX_PATH), help="索引输出路径")
    ap.add_argument("--chunk-size", type=int, default=None, help="切片大小（字符）")
    ap.add_argument("--overlap", type=int, default=None, help="片段重叠（字符）")
    ap.add_argument("--with-vectors", action="store_true", help="调用 Embedding API 生成向量")
    args = ap.parse_args()

    cfg = Config()
    chunk_size = args.chunk_size or cfg.chunk_size
    overlap = args.overlap if args.overlap is not None else cfg.chunk_overlap

    from pathlib import Path

    docs_dir = Path(args.docs)
    files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"未在 {docs_dir} 找到可解析文档（支持 {', '.join(sorted(SUPPORTED))}）")
        return 1

    t0 = time.perf_counter()
    chunks: list[dict] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        # 用文件名（去扩展名）作为来源标识，便于展示引用来源
        cs = build_chunks(text, f.stem, chunk_size, overlap)
        chunks.extend(cs)
        print(f"  {f.name:<32} → {len(cs):>3} 个片段")

    print(f"\n共 {len(files)} 份文档 → {len(chunks)} 个片段 "
          f"（chunk_size={chunk_size}, overlap={overlap}）")
    if chunks:
        avg = sum(c["chars"] for c in chunks) / len(chunks)
        print(f"平均片段长度：{avg:.0f} 字符")

    vectors: list[list[float]] = []
    if args.with_vectors:
        if not cfg.vector_enabled:
            print("\n[跳过向量化] 需在 .env 中配置 LLM_API_KEY / LLM_BASE_URL / LLM_EMBED_MODEL")
        else:
            from rag_core import LLMClient

            client = LLMClient(cfg)
            print(f"\n正在向量化（模型：{cfg.embed_model}）...")
            batch = 16
            for i in range(0, len(chunks), batch):
                part = chunks[i : i + batch]
                texts = [f"{c['heading']} {c['text']}" for c in part]
                vectors.extend(client.embed(texts))
                done = min(i + batch, len(chunks))
                print(f"  {done}/{len(chunks)}", end="\r", flush=True)
            print(f"  {len(chunks)}/{len(chunks)} 完成")

    payload = {"chunks": chunks, "vectors": vectors}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # 自检：加载一次，确认索引可用
    kb = KnowledgeBase(cfg, out).load()
    probe = kb.retrieve("请假两天找谁审批", 3)
    print(f"\n索引自检（问：请假两天找谁审批）：")
    for s in probe:
        print(f"  {s.score:.3f}  {s.label()}")
    print(f"\n索引已保存：{out}（耗时 {time.perf_counter() - t0:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
