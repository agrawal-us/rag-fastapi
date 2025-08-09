import os, json, pathlib, argparse, re
from typing import List, Dict, Tuple, Iterable

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rag_faiss import load_faiss, build_chain  # uses your existing code

# -----------------------------
# Utility: simple text cleanup
# -----------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

# -----------------------------
# Load gold queries
# -----------------------------
def load_gold(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # required keys: query, relevant (list of titles)
            obj["query"] = obj["query"].strip()
            obj["relevant"] = [t.strip() for t in obj.get("relevant", [])]
            items.append(obj)
    return items

# -----------------------------
# Retrieval metrics at k
# -----------------------------
def precision_recall_f1_at_k(
    gold: List[Dict],
    retrieved_titles: List[List[str]],
    k: int
) -> Tuple[float, float, float]:
    """
    gold[i]["relevant"] : list of titles that should be retrieved for query i
    retrieved_titles[i] : top-k titles returned by retriever for query i
    """
    assert len(gold) == len(retrieved_titles)
    precs, recs, f1s = [], [], []
    for i in range(len(gold)):
        gold_set = set(map(norm, gold[i]["relevant"]))
        got_set  = set(map(norm, retrieved_titles[i][:k]))
        tp = len(gold_set & got_set)
        fp = len(got_set - gold_set)
        fn = len(gold_set - got_set)

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0

        precs.append(prec); recs.append(rec); f1s.append(f1)
    return sum(precs)/len(precs), sum(recs)/len(recs), sum(f1s)/len(f1s)

# -----------------------------
# Faithfulness (two options)
# -----------------------------
def faithfulness_heuristic(answer: str, context_docs: List[Document]) -> float:
    """
    Cheap heuristic: proportion of answer sentences that share
    at least one 5+ char n-gram with the concatenated context.
    Not perfect, but works as a quick proxy without LLM judging.
    """
    import nltk  # optional: pip install nltk (only if you don’t already have it)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    ctx = norm(" ".join(d.page_content for d in context_docs))
    sents = [norm(s) for s in nltk.sent_tokenize(answer)]
    if not sents:
        return 0.0

    def supported(sent: str) -> bool:
        toks = [t for t in re.split(r"[^a-z0-9]+", sent) if len(t) >= 5]
        # any token >=5 chars present in context ⇒ weak support
        return any(t in ctx for t in toks)

    hits = sum(1 for s in sents if supported(s))
    return hits / len(sents)

def faithfulness_llm(answer: str, context_docs: List[Document], model: str = None) -> float:
    """
    LLM-as-judge: ask the model to score if the answer is grounded
    only in the provided context (0–1). Default uses your CHAT_MODEL.
    """
    model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    judge = ChatOpenAI(model=model, temperature=0)  # deterministic judge

    context = "\n---\n".join(d.page_content for d in context_docs[:6])  # cap context
    prompt = (
        "You are a strict evaluator of 'faithfulness'. "
        "Given a user answer and the provided context, score from 0.0 to 1.0 "
        "how well the answer is *fully supported* by the context without introducing unsupported claims.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Return ONLY a number between 0.0 and 1.0."
    )
    resp = judge.invoke(prompt)
    # extract float safely
    m = re.search(r"(\d+(?:\.\d+)?)", resp.content or "")
    try:
        v = float(m.group(1)) if m else 0.0
    except:
        v = 0.0
    return max(0.0, min(1.0, v))

# -----------------------------
# Run evaluation
# -----------------------------
def run_eval(
    gold_path: str,
    k: int = 4,
    judge: str = "heuristic"  # "heuristic" | "llm"
):
    # Load index + chain from your app
    db = load_faiss()
    retriever = db.as_retriever(search_kwargs={"k": k})
    chain = build_chain(retriever)

    gold = load_gold(gold_path)

    retrieved_titles_all: List[List[str]] = []
    faith_scores: List[float] = []

    for item in gold:
        q = item["query"]
        # 1) retrieval for metrics
        docs = retriever.get_relevant_documents(q)
        titles = [d.metadata.get("title", "") for d in docs]
        retrieved_titles_all.append(titles)

        # 2) generation + faithfulness
        out = chain.invoke(q)
        answer = out.content or ""

        if judge == "llm":
            faith = faithfulness_llm(answer, docs)
        else:
            faith = faithfulness_heuristic(answer, docs)
        faith_scores.append(faith)

    p, r, f1 = precision_recall_f1_at_k(gold, retrieved_titles_all, k)
    faith_avg = sum(faith_scores) / len(faith_scores)

    return {
        "k": k,
        "precision@k": round(p, 4),
        "recall@k": round(r, 4),
        "f1@k": round(f1, 4),
        "faithfulness": round(faith_avg, 4),
        "n": len(gold),
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG: Precision/Recall/F1 and Faithfulness")
    ap.add_argument("--gold", default="tests/gold.jsonl", help="Path to JSONL gold file")
    ap.add_argument("--k", type=int, default=4, help="Top-k for retrieval")
    ap.add_argument("--judge", choices=["heuristic","llm"], default="heuristic",
                    help="Faithfulness judge type")
    args = ap.parse_args()

    metrics = run_eval(args.gold, k=args.k, judge=args.judge)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()