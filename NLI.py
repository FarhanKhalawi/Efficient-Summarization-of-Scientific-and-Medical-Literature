# %% [markdown]
# Summarization + TF-IDF evaluation + Hallucination evaluation (NLI, numeric checks, unsupported terms)
#
# References:
# - TF-IDF / cosine: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#                     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# - NLI model (BART MNLI): https://huggingface.co/facebook/bart-large-mnli
# - Hallucination/factuality in summarization (why NLI > lexical overlap):
#   Maynez et al., 2020: https://aclanthology.org/2020.acl-main.173/
#   Pagnoni et al., 2021 (FRANK): https://aclanthology.org/2021.naacl-main.273/
#   Goyal & Durrett, 2021: https://aclanthology.org/2021.naacl-main.299/

# --- Set allocator env BEFORE importing torch ---
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # replaces deprecated PYTORCH_CUDA_ALLOC_CONF

import re
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("CUDA available:", torch.cuda.is_available())

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
CSV_PATH = "data/MeDAL/pretrain_subset/test.csv"

# bitsandbytes 4-bit config (nf4 + double quant + fp16 compute)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading tokenizer & model...")

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Ensure we have a pad token
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

# Load model (GPU path uses 4-bit quant + auto device map; CPU fallback allowed)
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype="auto",   # <- use dtype (torch_dtype is deprecated)
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

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def build_inputs(tok, system_msg, user_text):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text},
    ]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages, chat

def strip_all_think_blocks(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()

def safe_trim_to_first_n_sentences(text, n=4, min_keep=3):
    text = strip_all_think_blocks(text)
    text = re.sub(r"^\s*(?:Hmm[.,].*?\n+)+", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) > n:
        sents = sents[:n]
    if len(sents) < min_keep and len(text) > 0:
        return text
    return " ".join(sents).strip()

def postprocess(decoded: str) -> str:
    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1]
    m = re.search(r"<summary>(.*?)</summary>", decoded, flags=re.DOTALL | re.IGNORECASE)
    final = (m.group(1) if m else decoded).strip()
    if final.strip() in {"", "...", "…"}:
        return ""
    return safe_trim_to_first_n_sentences(final, n=4, min_keep=3)

def sanitize_numeric_glitches(text: str) -> str:
    # Fix patterns like "n=-3" -> "n=3", and "- 6 sheep" -> "6 sheep"
    text = re.sub(r'(?i)(n=\s*)-\s*(\d+)', r'\1\2', text)
    text = re.sub(r'(?<!\d)-\s*(\d+)\s+(sheep|patients?|subjects?)', r'\1 \2', text, flags=re.I)
    return text

def get_embedding_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device

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

STOP_STRINGS = ["</summary>"]
STOPPER = StoppingCriteriaList([StopOnSubstrings(STOP_STRINGS, tok)])

def cut_after_last_summary_tag(text: str) -> str:
    end_tag = "</summary>"
    pos = text.find(end_tag)
    return text[:pos + len(end_tag)] if pos != -1 else text

def get_model_ctx(model, default_ctx=262144):
    ctx = getattr(getattr(model, "config", object()), "max_position_embeddings", None)
    return int(ctx) if isinstance(ctx, int) and ctx > 0 else default_ctx

def generate_summary(model, tok, chat, system_msg_for_rebuild,
                     keep_tokens_for_answer=512, source_text=None,
                     gen_temp=0.3, gen_top_p=0.9, greedy=False):
    max_ctx = int(get_model_ctx(model))
    reserve = keep_tokens_for_answer + 64
    prompt_budget = max(256, max_ctx - reserve)

    inputs = tok([chat], return_tensors="pt", truncation=True, max_length=prompt_budget)
    inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}

    if inputs["input_ids"].shape[1] >= prompt_budget and source_text:
        head_enc = tok(source_text, return_tensors="pt", truncation=True, max_length=prompt_budget)
        trimmed = tok.decode(head_enc["input_ids"][0], skip_special_tokens=True)
        _, chat2 = build_inputs(tok, system_msg_for_rebuild, trimmed)
        inputs = tok([chat2], return_tensors="pt", truncation=True, max_length=prompt_budget)
        inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    gen_kwargs = dict(
        max_new_tokens=320,
        eos_token_id=tok.eos_token_id,
        stopping_criteria=STOPPER,
        no_repeat_ngram_size=3,
    )
    if greedy:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=gen_temp, top_p=gen_top_p))

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded = cut_after_last_summary_tag(decoded)
    return decoded

# -----------------------------------------------------------------------------
# Data load (CSV or fallback)
# -----------------------------------------------------------------------------
try:
    df = pd.read_csv(CSV_PATH)
    source_text = str(df.iloc[0]["TEXT"]).strip()
    if not source_text:
        raise ValueError("Empty TEXT cell in CSV.")
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
    print("Using fallback abstract. Characters:", len(source_text))

# -----------------------------------------------------------------------------
# Prompts & generate
# -----------------------------------------------------------------------------
SYSTEM_MSG_TAGGED = (
    "Summarize the user's medical abstract in 3–4 sentences. "
    "Be clear and factual. Keep key clinical details (condition, intervention, measurements, outcomes). "
    "Copy numeric values (n, percentages, measurements) exactly from the input; omit if unspecified. "
    "Return ONLY the summary wrapped exactly as:\n<summary>...</summary>\nNo preface, no analysis, no extra text."
)

SYSTEM_MSG_FALLBACK = (
    "Summarize the user's medical abstract in 3–4 sentences. "
    "Be clear and factual. Keep key clinical details (condition, intervention, measurements, outcomes). "
    "Copy numeric values exactly; omit if unspecified. "
    "Output ONLY the 3–4 sentence summary—no preface, no analysis."
)

GEN_TEMP, GEN_TOPP = 0.3, 0.9

# Attempt 1
_, chat = build_inputs(tok, SYSTEM_MSG_TAGGED, source_text)
print("Generating summary (attempt 1)...")
decoded = generate_summary(
    model, tok, chat, system_msg_for_rebuild=SYSTEM_MSG_TAGGED,
    keep_tokens_for_answer=512, source_text=source_text,
    gen_temp=GEN_TEMP, gen_top_p=GEN_TOPP, greedy=False
)
summary = postprocess(decoded)
summary = sanitize_numeric_glitches(summary)

# Fallback: deterministic pass
if not summary:
    print("Generating summary (attempt 2, deterministic fallback)...")
    _, chat_fb = build_inputs(tok, SYSTEM_MSG_FALLBACK, source_text)
    decoded_fb = generate_summary(
        model, tok, chat_fb, system_msg_for_rebuild=SYSTEM_MSG_FALLBACK,
        keep_tokens_for_answer=512, source_text=source_text,
        gen_temp=0.2, gen_top_p=0.95, greedy=True
    )
    summary = postprocess(decoded_fb)
    summary = sanitize_numeric_glitches(summary)

if not summary:
    preview = (decoded or "")[:400].replace("\n", " ")
    print("\n[Debug] Model raw (first 400 chars):", preview)
    summary = "Summary unavailable: the model did not produce a clean summary."

print("\n--- SUMMARY ---\n")
print(summary)

os.makedirs("outputs", exist_ok=True)
out_path = "outputs/summary_first_test.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(summary + "\n")
print(f"\nSaved to {out_path}")

# -----------------------------------------------------------------------------
# TF-IDF evaluation (robust for tiny corpora)
# -----------------------------------------------------------------------------
def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def _safe_max_df(n_docs: int, max_df):
    if isinstance(max_df, float) and n_docs <= 10:
        return 1.0
    return max_df

def tfidf_eval(source_text: str, summary: str,
               ngram_range=(1,2), max_df=0.9, min_df=1,
               coverage_threshold=0.10, use_stopwords=True):
    stop = 'english' if use_stopwords else None

    # Global similarity (2 docs)
    docs = [source_text, summary]
    vec_global = TfidfVectorizer(lowercase=True, stop_words=stop,
                                 ngram_range=ngram_range,
                                 max_df=_safe_max_df(len(docs), max_df),
                                 min_df=min_df)
    X = vec_global.fit_transform(docs)
    global_sim = float(cosine_similarity(X[0], X[1])[0, 0])

    # Coverage (source sentences vs summary)
    src_sents = split_sentences(source_text)[:200] or [source_text]
    cov_docs = src_sents + [summary]
    vec_cov = TfidfVectorizer(lowercase=True, stop_words=stop,
                              ngram_range=ngram_range,
                              max_df=_safe_max_df(len(cov_docs), max_df),
                              min_df=min_df)
    X_cov = vec_cov.fit_transform(cov_docs)
    S, q = X_cov[:-1], X_cov[-1]
    sent_sims = cosine_similarity(S, q).ravel()
    coverage = float((sent_sims >= coverage_threshold).mean())

    # Redundancy (within-summary)
    summ_sents = split_sentences(summary)
    if len(summ_sents) >= 2:
        vec_red = TfidfVectorizer(lowercase=True, stop_words=stop,
                                  ngram_range=ngram_range,
                                  max_df=1.0, min_df=1)
        X_red = vec_red.fit_transform(summ_sents)
        C = cosine_similarity(X_red)
        redundancy = float((C.sum() - np.trace(C)) / (C.shape[0]*C.shape[1] - C.shape[0]))
    else:
        redundancy = 0.0

    # Top keywords (summary)
    top_keywords = []
    if len(re.findall(r"\w+", summary)) >= 2:
        vec_kw = TfidfVectorizer(lowercase=True, stop_words=stop,
                                 ngram_range=ngram_range,
                                 max_df=1.0, min_df=1)
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
        f"coverage@{coverage_threshold:.2f}": coverage,
        "redundancy_avg_pairwise": redundancy,
        "top_keywords": top_keywords,
        "notes": "Higher similarity & coverage are good; lower redundancy is better."
    }

metrics = tfidf_eval(
    source_text,
    summary,
    ngram_range=(1,2),
    max_df=0.9,
    min_df=1,
    coverage_threshold=0.10,
    use_stopwords=True
)
print("\n--- TF-IDF EVAL ---")
for k, v in metrics.items():
    if k == "top_keywords":
        print(f"{k}: {[w for w,_ in v]}")
    else:
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

with open("outputs/tfidf_eval.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("Saved TF-IDF metrics to outputs/tfidf_eval.json")

# -----------------------------------------------------------------------------
# Hallucination evaluation: NLI + numeric consistency + unsupported terms
# -----------------------------------------------------------------------------
def extract_numbers(text: str):
    # captures ints/decimals/percents, forms like "n=12", "12%", "18.5", "1,200"
    nums = re.findall(r'(?:n\s*=\s*)?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?', text, flags=re.I)
    norm = []
    for x in nums:
        y = x.lower().replace(' ', '')
        y = y.replace(',', '')  # 1,200 -> 1200
        norm.append(y)
    return norm

def content_words(text: str):
    stop = set("""
        a an the and or of in on with without to for from by as at is are was were be been being that this these those
        it its their his her them they we you i he she not no yes do does did done than then over under into out up down
    """.split())
    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-']+", text)]
    return [t for t in toks if len(t) > 2 and t not in stop]

def hallucination_eval(source_text: str, summary: str, device=None):
    # 1) NLI-based factuality per sentence (premise=source, hypothesis=sent)
    nli_dev = (device if device is not None else (0 if torch.cuda.is_available() else -1))

    nli = pipeline(
        "text-classification",
        model="facebook/bart-large-mnli",
        device=nli_dev,
        return_all_scores=True   # get ENTAILMENT / NEUTRAL / CONTRADICTION scores
    )

    summ_sents = split_sentences(summary) or [summary]

    entail_scores, contra_scores, neutral_scores = [], [], []
    for sent in summ_sents:
        # For MNLI, pass: text=premise, text_pair=hypothesis
        out = nli({"text": source_text, "text_pair": sent}, function_to_apply="softmax")[0]
        score_map = {o["label"].lower(): float(o["score"]) for o in out}
        entail_scores.append(score_map.get("entailment", 0.0))
        contra_scores.append(score_map.get("contradiction", 0.0))
        neutral_scores.append(score_map.get("neutral", 0.0))

    entail_rate = float(np.mean([s > 0.5 for s in entail_scores]))
    contra_rate = float(np.mean([s > 0.5 for s in contra_scores]))
    avg_entail = float(np.mean(entail_scores))
    avg_contra = float(np.mean(contra_scores))

    # 2) Numeric consistency: numbers in summary not present in source
    src_nums = set(extract_numbers(source_text))
    sum_nums = extract_numbers(summary)
    numeric_unsupported = [n for n in sum_nums if n not in src_nums]

    # 3) Unsupported-term rate: summary content words missing from source
    src_vocab = set(content_words(source_text))
    sum_vocab = content_words(summary)
    missing_terms = [w for w in sum_vocab if w not in src_vocab]
    unsupported_term_rate = float(len(missing_terms) / max(1, len(sum_vocab)))

    # Simple combined heuristic flag (tune thresholds for your domain)
    hallucination_flag = (contra_rate > 0.2) or (len(numeric_unsupported) > 0) or (unsupported_term_rate > 0.5)

    return {
        "nli_avg_entailment": avg_entail,
        "nli_avg_contradiction": avg_contra,
        "nli_entailment_rate@0.5": entail_rate,
        "nli_contradiction_rate@0.5": contra_rate,
        "numeric_unsupported": numeric_unsupported,
        "unsupported_term_rate": unsupported_term_rate,
        "missing_terms_sample": missing_terms[:20],
        "hallucination_flag": hallucination_flag,
        "notes": "Lower contradiction, empty numeric_unsupported, and low unsupported_term_rate indicate fewer hallucinations.",
        "per_sentence_entailment": dict(zip(summ_sents, [float(s) for s in entail_scores])),
        "per_sentence_contradiction": dict(zip(summ_sents, [float(s) for s in contra_scores])),
    }

# Important: free the big model before creating the NLI pipeline to avoid OOM
del model
torch.cuda.empty_cache()

hall = hallucination_eval(source_text, summary, device=None)  # set device=-1 to force CPU
print("\n--- HALLUCINATION EVAL ---")
for k, v in hall.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    elif isinstance(v, list):
        print(f"{k}: {v}")
    elif isinstance(v, dict):
        print(f"{k}:")
        for kk, vv in v.items():
            try:
                print(f"  - {kk} :: {vv:.3f}")
            except Exception:
                print(f"  - {kk} :: {vv}")
    else:
        print(f"{k}: {v}")

with open("outputs/hallucination_eval.json", "w", encoding="utf-8") as f:
    json.dump(hall, f, ensure_ascii=False, indent=2)
print("Saved hallucination metrics to outputs/hallucination_eval.json")
