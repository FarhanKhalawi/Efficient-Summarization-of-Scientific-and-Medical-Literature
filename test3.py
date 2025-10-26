# -*- coding: utf-8 -*-
# =============================================================================
# Summarization + TF-IDF diagnostics + ROUGE evaluation (model vs. human refs)
#
# References (official docs/papers & source code):
# - scikit-learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# - scikit-learn cosine_similarity: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# - rouge-score (Google Research) docs/source: https://pypi.org/project/rouge-score/ , https://github.com/google-research/google-research/tree/master/rouge
# - ROUGE paper: Lin, C.-Y. (2004) https://aclanthology.org/W04-1013/
# =============================================================================

# ---------- Set allocator env BEFORE importing torch ----------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # new name

import re, json, numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer, scoring

print("CUDA available:", torch.cuda.is_available())

# ---------- Paths ----------
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
CSV_PATH = "data/MeDAL/pretrain_subset/test.csv"
HUMAN_CSV = "data/MeDAL/pretrain_subset/human_summaries_for_rouge.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 4-bit quant config ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading tokenizer & model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Ensure pad token
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

# Load model
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype="auto",              # replaces deprecated torch_dtype
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_folder="offload",
        max_memory={0: "30GiB", "cpu": "64GiB"},
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_folder="offload",
        max_memory={"cpu": "64GiB"},
    )

model.eval()
print("Model loaded successfully!")

# ---------- Helpers ----------
def build_inputs(tok, system_msg, user_text):
    messages = [{"role":"system","content":system_msg},
                {"role":"user","content":user_text}]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat

def strip_all_think_blocks(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()

def safe_trim_to_first_n_sentences(text, n=4, min_keep=3):
    text = strip_all_think_blocks(text)
    text = re.sub(r"^\s*(?:Hmm[.,].*?\n+)+", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) > n: sents = sents[:n]
    if len(sents) < min_keep and len(text) > 0: return text
    return " ".join(sents).strip()

def postprocess(decoded: str) -> str:
    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1]
    m = re.search(r"<summary>(.*?)</summary>", decoded, flags=re.DOTALL | re.IGNORECASE)
    final = (m.group(1) if m else decoded).strip()
    if final.strip() in {"", "...", "…"}: return ""
    return safe_trim_to_first_n_sentences(final, n=4, min_keep=3)

def sanitize_numeric_glitches(text: str) -> str:
    text = re.sub(r'(?i)(n=\s*)-\s*(\d+)', r'\1\2', text)
    text = re.sub(r'(?<!\d)-\s*(\d+)\s+(sheep|patients?|subjects?)', r'\1 \2', text, flags=re.I)
    return text

def get_embedding_device(model):
    try: return model.get_input_embeddings().weight.device
    except Exception: return next(model.parameters()).device

EMBED_DEVICE = get_embedding_device(model)
print("Embedding device:", EMBED_DEVICE)

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        super().__init__()
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_tokens = input_ids[0].tolist()
        for s_ids in self.stop_ids:
            if len(last_tokens) >= len(s_ids) and last_tokens[-len(s_ids):] == s_ids:
                return True
        return False

STOPPER = StoppingCriteriaList([StopOnSubstrings(["</summary>"], tok)])

def cut_after_last_summary_tag(text: str) -> str:
    pos = text.find("</summary>")
    return text[:pos+len("</summary>")] if pos != -1 else text

def get_model_ctx(model, default_ctx=262144):
    ctx = getattr(getattr(model, "config", object()), "max_position_embeddings", None)
    return int(ctx) if isinstance(ctx, int) and ctx > 0 else default_ctx

def generate_summary(model, tok, chat,
                     keep_tokens_for_answer=512, source_text=None,
                     gen_temp=0.3, gen_top_p=0.9, greedy=False):
    max_ctx = int(get_model_ctx(model))
    prompt_budget = max(256, max_ctx - (keep_tokens_for_answer + 64))
    inputs = tok([chat], return_tensors="pt", truncation=True, max_length=prompt_budget)
    inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    gen_kwargs = dict(
        max_new_tokens=320,
        eos_token_id=tok.eos_token_id,
        stopping_criteria=STOPPER,
        no_repeat_ngram_size=3,
    )
    if greedy: gen_kwargs.update(dict(do_sample=False))
    else:      gen_kwargs.update(dict(do_sample=True, temperature=gen_temp, top_p=gen_top_p))
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return cut_after_last_summary_tag(decoded)

def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', str(text)) if s.strip()]

# ---------- Load data (CSV or fallback) ----------
try:
    df = pd.read_csv(CSV_PATH)
    source_text = str(df.iloc[0]["TEXT"]).strip()
    # capture ABSTRACT_ID if present for alignment with human references
    abs_id = int(df.iloc[0]["ABSTRACT_ID"]) if "ABSTRACT_ID" in df.columns else None
    print("Loaded CSV sample. Characters:", len(source_text))
except Exception as e:
    print("CSV load warning:", e)
    source_text = (
        "Background: Hypertension is a common cardiovascular risk factor. "
        "Methods: We conducted a randomized, controlled trial evaluating a new ACE inhibitor versus placebo "
        "in 1,200 adults with stage 2 hypertension over 24 weeks. "
        "Results: The treatment group showed a mean systolic BP reduction of 18 mmHg versus 6 mmHg with placebo; "
        "adverse events were mild and comparable. "
        "Conclusion: The ACE inhibitor significantly reduced blood pressure with acceptable safety."
    )
    abs_id = None
    print("Using fallback abstract. Characters:", len(source_text))

# ---------- Prompts & generate ----------
SYSTEM_MSG_TAGGED = (
    "Summarize the user's medical abstract in 3–4 sentences. "
    "Be clear and factual. Keep key clinical details (condition, intervention, measurements, outcomes). "
    "Copy numeric values (n, percentages, measurements) exactly from the input; omit if unspecified. "
    "Return ONLY the summary wrapped exactly as:\n<summary>...</summary>\nNo preface, no analysis, no extra text."
)
chat = build_inputs(tok, SYSTEM_MSG_TAGGED, source_text)
print("Generating summary (attempt 1)...")
decoded = generate_summary(model, tok, chat, keep_tokens_for_answer=512, source_text=source_text,
                           gen_temp=0.3, gen_top_p=0.9, greedy=False)
summary = sanitize_numeric_glitches(postprocess(decoded))

if not summary:
    print("Attempt 1 empty — trying deterministic fallback...")
    SYSTEM_MSG_FALLBACK = (
        "Summarize the user's medical abstract in 3–4 sentences. "
        "Be clear and factual. Keep key clinical details. Copy numeric values exactly; omit if unspecified. "
        "Output ONLY the 3–4 sentence summary—no preface."
    )
    chat2 = build_inputs(tok, SYSTEM_MSG_FALLBACK, source_text)
    decoded2 = generate_summary(model, tok, chat2, keep_tokens_for_answer=512,
                                source_text=source_text, gen_temp=0.2, gen_top_p=0.95, greedy=True)
    summary = sanitize_numeric_glitches(postprocess(decoded2)) or "Summary unavailable."

print("\n--- SUMMARY ---\n")
print(summary)

# save summary
with open(os.path.join(OUT_DIR, "summary_first_test.txt"), "w", encoding="utf-8") as f:
    f.write(summary + "\n")
print("\nSaved to outputs/summary_first_test.txt")

# ========================= TF-IDF diagnostics ===============================
def _safe_max_df(n_docs: int, max_df):
    return 1.0 if isinstance(max_df, float) and n_docs <= 10 else max_df

def tfidf_eval(source_text: str, summary: str,
               ngram_range=(1,2), max_df=0.9, min_df=1,
               coverage_threshold=0.10, use_stopwords=True):
    stop = 'english' if use_stopwords else None
    # Global similarity
    docs = [source_text, summary]
    vec_global = TfidfVectorizer(lowercase=True, stop_words=stop,
                                 ngram_range=ngram_range, max_df=_safe_max_df(len(docs), max_df), min_df=min_df)
    X = vec_global.fit_transform(docs)
    global_sim = float(cosine_similarity(X[0], X[1])[0, 0])
    # Coverage
    src_sents = split_sentences(source_text)[:200] or [source_text]
    cov_docs = src_sents + [summary]
    vec_cov = TfidfVectorizer(lowercase=True, stop_words=stop,
                              ngram_range=ngram_range, max_df=_safe_max_df(len(cov_docs), max_df), min_df=min_df)
    X_cov = vec_cov.fit_transform(cov_docs)
    S, q = X_cov[:-1], X_cov[-1]
    sent_sims = cosine_similarity(S, q).ravel()
    coverage = float((sent_sims >= coverage_threshold).mean())
    # Redundancy
    summ_sents = split_sentences(summary)
    if len(summ_sents) >= 2:
        vec_red = TfidfVectorizer(lowercase=True, stop_words=stop, ngram_range=ngram_range, max_df=1.0, min_df=1)
        X_red = vec_red.fit_transform(summ_sents)
        C = cosine_similarity(X_red)
        redundancy = float((C.sum() - np.trace(C)) / (C.shape[0]*C.shape[1] - C.shape[0]))
    else:
        redundancy = 0.0
    # Keywords
    top_keywords = []
    if len(re.findall(r"\w+", summary)) >= 2:
        vec_kw = TfidfVectorizer(lowercase=True, stop_words=stop, ngram_range=ngram_range, max_df=1.0, min_df=1)
        try:
            X_kw = vec_kw.fit_transform([summary])
            vocab = np.array(vec_kw.get_feature_names_out())
            scores = X_kw.toarray()[0]
            top_idx = scores.argsort()[::-1][:10]
            top_keywords = [(vocab[i], float(scores[i])) for i in top_idx if scores[i] > 0]
        except ValueError:
            top_keywords = []
    return {
        "tfidf_cosine_similarity": global_sim,
        "coverage@{:.2f}".format(coverage_threshold): coverage,
        "redundancy_avg_pairwise": redundancy,
        "top_keywords": top_keywords
    }

metrics = tfidf_eval(source_text, summary, ngram_range=(1,2), coverage_threshold=0.10)
print("\n--- TF-IDF EVAL ---")
for k, v in metrics.items():
    if k == "top_keywords": print(f"{k}: {[w for w,_ in v]}")
    else: print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

with open(os.path.join(OUT_DIR, "tfidf_eval.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("Saved TF-IDF metrics to outputs/tfidf_eval.json")

# ============================ ROUGE evaluation ==============================
def _find_ref_col(df_refs: pd.DataFrame) -> str:
    for c in ["HUMAN_SUMMARY", "SUMMARY", "reference", "REF"]:
        if c in df_refs.columns:
            return c
    raise ValueError(
        "Could not find a reference-summary column in HUMAN_CSV. "
        "Expected one of: HUMAN_SUMMARY, SUMMARY, reference, REF."
    )

def _to_plain_text(t: str) -> str:
    # light cleanse, keep content (don’t over-normalize for ROUGE)
    t = strip_all_think_blocks(str(t))
    t = re.sub(r"</?summary>", "", t, flags=re.I)
    return t.strip()

def _load_human_reference(human_csv_path: str, abstract_id):
    hdf = pd.read_csv(human_csv_path)
    ref_col = _find_ref_col(hdf)

    # If we have ABSTRACT_ID in both places, align on it
    if abstract_id is not None and "ABSTRACT_ID" in hdf.columns:
        hit = hdf.loc[hdf["ABSTRACT_ID"] == abstract_id]
        if not hit.empty:
            refs = [str(r).strip() for r in hit[ref_col].dropna().tolist() if str(r).strip()]
            if refs:
                return refs

    # Fallback: use *all* non-empty references in the file (multi-ref evaluation)
    refs = [str(r).strip() for r in hdf[ref_col].dropna().tolist() if str(r).strip()]
    if not refs:
        raise ValueError("No non-empty human reference summaries found in HUMAN_CSV.")
    return refs

print("\n--- ROUGE EVAL ---")
try:
    references = _load_human_reference(HUMAN_CSV, abs_id)
    pred = _to_plain_text(summary)
    refs_plain = [_to_plain_text(r) for r in references]

    # Use stemming and the ROUGE-Lsum variant recommended for summaries
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    # score_multi supports multiple references
    if len(refs_plain) > 1:
        s = scorer.score_multi(refs_plain, pred)  # dict of Score objects
    else:
        s = scorer.score(refs_plain[0], pred)

    # Save per-sample scores
    per_sample_rows = []
    def _row_from_score(tag, scr):
        return {
            "metric": tag,
            "precision": float(scr.precision),
            "recall": float(scr.recall),
            "f1": float(scr.fmeasure),
        }
    for tag, scr in s.items():
        per_sample_rows.append(_row_from_score(tag, scr))

    # Aggregate with bootstrap (works for corpora; still returns CI for single item)
    agg = scoring.BootstrapAggregator()
    agg.add_scores(s)
    agg_res = agg.aggregate()

    for tag in ["rouge1", "rouge2", "rougeLsum"]:
        mid = agg_res[tag].mid
        print(f"{tag.upper():9s} F1={mid.fmeasure:.4f}")

    with open(os.path.join(OUT_DIR, "rouge_per_sample.json"), "w", encoding="utf-8") as f:
        json.dump(per_sample_rows, f, ensure_ascii=False, indent=2)

    agg_out = {
        k: {
            "precision": float(v.mid.precision),
            "recall": float(v.mid.recall),
            "f1": float(v.mid.fmeasure),
            "low_f1": float(v.low.fmeasure),
            "high_f1": float(v.high.fmeasure),
        }
        for k, v in agg_res.items()
    }
    with open(os.path.join(OUT_DIR, "rouge_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(agg_out, f, ensure_ascii=False, indent=2)

    print("Saved ROUGE results to outputs/rouge_per_sample.json and outputs/rouge_aggregate.json")

except Exception as e:
    print("ROUGE evaluation error:", e)
