import os
import json
import argparse
import pathlib
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# LangChain / OpenAI wrappers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# Our local RAG bits
from rag_faiss import load_faiss, build_chain  # reuses your existing index + chain builder

# RAGAS
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset


def _log(msg: str):
    print(f"[ragas] {msg}")


def load_gold_any(path: str) -> List[Dict]:
    """
    Load a JSONL gold file. Supports two styles:
      1) Retrieval-only: {"query": "...", "relevant": ["titleA", "titleB"]}
      2) Generation gold: {"query": "...", "expected": "reference answer"}
         - 'expected' becomes ground_truth for RAGAS (wrapped as a 1-item list).
    """
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query", "").strip()
            expected = obj.get("expected", None)
            relevant = obj.get("relevant", [])
            rows.append({"query": q, "expected": expected, "relevant": relevant})
    return rows


def run_ragas_eval(gold_path: str, k: int = 4) -> Dict:
    """
    Build a RAGAS dataset by running your retriever + generator for each gold query, then
    compute RAGAS metrics. If the gold file contains 'expected', we also include
    answer_correctness. Otherwise, we skip it.
    """
    # Load FAISS + chain from your app
    db = load_faiss()
    retriever = db.as_retriever(search_kwargs={"k": k})
    chain = build_chain(retriever)

    rows = load_gold_any(gold_path)

    questions: List[str] = []
    contexts: List[List[str]] = []
    answers: List[str] = []
    references: List[str] = []
    ground_truths: List[List[str]] = []

    has_ground_truth = any(
        (r.get("expected") is not None) and str(r.get("expected")).strip() != ""
        for r in rows
    )

    for r in rows:
        q = r["query"]
        # retrieve
        docs: List[Document] = retriever.invoke(q)
        ctx_texts = [d.page_content for d in docs]

        # generate
        out = chain.invoke(q)
        ans = (out.content or "").strip() if hasattr(out, "content") else str(out).strip()

        questions.append(q)
        contexts.append(ctx_texts)
        answers.append(ans)
        gt = r.get("expected")
        ref = str(gt).strip() if (gt is not None and str(gt).strip() != "") else ""
        references.append(ref)
        # Keep both fields for RAGAS version compatibility:
        # - 'reference' as a string (may be "")
        # - 'ground_truth' as a list[str] (use [""] when empty)
        ground_truths.append([ref] if ref != "" else [""])

    data = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "reference": references,       # string per sample ("" if unknown)
        "ground_truth": ground_truths, # list[str], kept for compatibility
    }
    ds = Dataset.from_dict(data)

    # LLM + embeddings for RAGAS
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    llm = ChatOpenAI(model=chat_model, temperature=0)
    emb = OpenAIEmbeddings(model=embed_model)

    # Choose metrics; include answer_correctness only if we have ground truth
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    if has_ground_truth:
        metrics.append(answer_correctness)

    _log(f"Evaluating {len(questions)} samples at k={k} using metrics: "
         + ", ".join(m.name for m in metrics))

    result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb)

    # Convert to a compact summary + show per-sample path for debugging
    df = result.to_pandas()
    summary = {
        col: float(df[col].mean())
        for col in df.columns
        if col not in ("question", "answer", "contexts", "ground_truth", "reference")
    }
    summary["n"] = len(df)

    return {"summary": summary, "details": df.to_dict(orient="records")}


def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG generation with RAGAS")
    ap.add_argument("--gold", default="tests/gold_gen.jsonl",
                    help="Path to JSONL gold file. Supports 'expected' for ground truth.")
    ap.add_argument("--k", type=int, default=4, help="Top-k for retrieval")
    ap.add_argument("--out", type=str, default="ragas_report.json",
                    help="Write a JSON report with summary and per-sample details")
    args = ap.parse_args()

    report = run_ragas_eval(args.gold, k=args.k)

    # Print summary nicely
    print(json.dumps(report["summary"], indent=2))

    # Persist full report
    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"Wrote report → {out_path.resolve()}")


if __name__ == "__main__":
    main()