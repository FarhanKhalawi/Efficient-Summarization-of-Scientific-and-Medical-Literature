# %% [markdown]
# Summarization notebook — robust </summary> stopping, safer decoding, EOS stop, deterministic fallback

# %%


import os, re, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList

# Mitigate CUDA allocator fragmentation (must be set before CUDA allocations)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_folder="offload",
        max_memory={0: "30GiB", "cpu": "64GiB"},
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        offload_folder="offload",
        max_memory={"cpu": "64GiB"},
    )

model.eval()
print("Model loaded successfully!")

# %%
def build_inputs(tok, system_msg, user_text):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text},
    ]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages, chat

def strip_all_think_blocks(t: str) -> str:
    # Remove any nested <think>...</think> sections robustly
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()

def safe_trim_to_first_n_sentences(text, n=4, min_keep=3):
    text = strip_all_think_blocks(text)
    # Also remove any leading "Hmm," style prefaces
    text = re.sub(r"^\s*(?:Hmm[.,].*?\n+)+", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) > n:
        sents = sents[:n]
    if len(sents) < min_keep and len(text) > 0:
        return text
    return " ".join(sents).strip()

def postprocess(decoded: str) -> str:
    # Keep only the text after the LAST </think> (if any slipped through)
    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1]
    # If <summary>...</summary> present, extract the inner text
    m = re.search(r"<summary>(.*?)</summary>", decoded, flags=re.DOTALL | re.IGNORECASE)
    final = (m.group(1) if m else decoded).strip()
    # Normalize empty/ellipsis
    if final.strip() in {"", "...", "…"}:
        return ""
    return safe_trim_to_first_n_sentences(final, n=4, min_keep=3)

def get_embedding_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device

EMBED_DEVICE = get_embedding_device(model)
print("Embedding device:", EMBED_DEVICE)

class StopOnSubstrings(StoppingCriteria):
    """
    Stop generation once any of the provided stop strings appears at the end (token-wise).
    This is complemented by a decoded-text cut for robustness.
    """
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

# We REMOVE bad_words_ids (it risks corrupting legitimate words).
# We'll rely on instruction design + post-filtering instead.

def cut_after_last_summary_tag(text: str) -> str:
    """
    Hard-cut decoded text right after the first occurrence of </summary>.
    This complements token-level stopping and ensures a clean end.
    """
    end_tag = "</summary>"
    pos = text.find(end_tag)
    if pos != -1:
        return text[:pos + len(end_tag)]
    return text

def get_model_ctx(model, default_ctx=262144):
    """
    Try to read the model's max context length; fall back to a large safe default.
    """
    ctx = getattr(getattr(model, "config", object()), "max_position_embeddings", None)
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    return default_ctx  # Qwen3 models advertise very long contexts

def generate_summary(model, tok, chat, system_msg_for_rebuild,
                     keep_tokens_for_answer=512, source_text=None,
                     gen_temp=0.4, gen_top_p=0.9, greedy=False):
    # Determine context and set a prompt budget
    max_ctx = int(get_model_ctx(model))
    reserve = keep_tokens_for_answer + 64
    prompt_budget = max(256, max_ctx - reserve)

    # Tokenize + move to embedding device
    inputs = tok([chat], return_tensors="pt", truncation=True, max_length=prompt_budget)
    inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}

    # Rebuild from head if still too long
    if inputs["input_ids"].shape[1] >= prompt_budget and source_text:
        head_enc = tok(source_text, return_tensors="pt", truncation=True, max_length=prompt_budget)
        trimmed = tok.decode(head_enc["input_ids"][0], skip_special_tokens=True)
        _, chat2 = build_inputs(tok, system_msg_for_rebuild, trimmed)
        inputs = tok([chat2], return_tensors="pt", truncation=True, max_length=prompt_budget)
        inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}

    # Ensure pad token again (defensive)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    gen_kwargs = dict(
        max_new_tokens=320,
        eos_token_id=tok.eos_token_id,     # allow EOS to stop as well
        stopping_criteria=STOPPER,         # primary hard stop at </summary>
        no_repeat_ngram_size=3,
    )
    if greedy:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=gen_temp, top_p=gen_top_p))

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # Hard-cut after </summary> in decoded space for robustness
    decoded = cut_after_last_summary_tag(decoded)
    return decoded

# %%
# Try to load a CSV sample; fall back to a small synthetic abstract for demo
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

# %%
SYSTEM_MSG_TAGGED = (
    "Summarize the user's medical abstract in 3–4 sentences. "
    "Be clear and factual. Keep key clinical details (condition, intervention, measurements, outcomes). "
    "Return ONLY the summary wrapped exactly as:\n<summary>...</summary>\nNo preface, no analysis, no extra text."
)

SYSTEM_MSG_FALLBACK = (
    "Summarize the user's medical abstract in 3–4 sentences. "
    "Be clear and factual. Keep key clinical details (condition, intervention, measurements, outcomes). "
    "Output ONLY the 3–4 sentence summary—no preface, no analysis."
)

GEN_TEMP, GEN_TOPP = 0.4, 0.9

# Attempt 1: tagged prompt that must end with </summary>
_, chat = build_inputs(tok, SYSTEM_MSG_TAGGED, source_text)
print("Generating summary (attempt 1)...")
decoded = generate_summary(
    model, tok, chat, system_msg_for_rebuild=SYSTEM_MSG_TAGGED,
    keep_tokens_for_answer=512, source_text=source_text,
    gen_temp=GEN_TEMP, gen_top_p=GEN_TOPP, greedy=False
)
summary = postprocess(decoded)

# Fallback: deterministic / greedy pass (often more stable when the first try fails)
if not summary:
    print("Generating summary (attempt 2, deterministic fallback)...")
    _, chat_fb = build_inputs(tok, SYSTEM_MSG_FALLBACK, source_text)
    decoded_fb = generate_summary(
        model, tok, chat_fb, system_msg_for_rebuild=SYSTEM_MSG_FALLBACK,
        keep_tokens_for_answer=512, source_text=source_text,
        gen_temp=0.7, gen_top_p=0.95, greedy=True
    )
    summary = postprocess(decoded_fb)

# Final guardrail: if the model STILL tries meta commentary, strip any leading lone "I ..." lines
if summary and summary[:1].lower() == "i":
    summary = re.sub(r"(^|\n)I[^\n]*", "", summary).strip()

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



# --- TF-IDF evaluation add-on ---
import re, json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_sentences(text: str):
    # lightweight sentence split (works fine for abstracts)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def tfidf_eval(source_text: str, summary: str,
              ngram_range=(1,2), max_df=0.9, min_df=1,
              coverage_threshold=0.10):
    # Global similarity (source vs summary)
    vec_global = TfidfVectorizer(lowercase=True, stop_words='english',
                                 ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    X = vec_global.fit_transform([source_text, summary])
    global_sim = float(cosine_similarity(X[0], X[1])[0, 0])

    # Sentence-level coverage (source sentences “matched” by summary content)
    src_sents = split_sentences(source_text)[:200]  # cap to avoid huge docs
    if not src_sents:
        src_sents = [source_text]
    vec_cov = TfidfVectorizer(lowercase=True, stop_words='english',
                              ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    X_cov = vec_cov.fit_transform(src_sents + [summary])
    S = X_cov[:-1]         # source sentences
    q = X_cov[-1]          # summary
    sent_sims = cosine_similarity(S, q).ravel()
    coverage = float((sent_sims >= coverage_threshold).mean())

    # Redundancy: average pairwise cosine among summary sentences (lower is better)
    summ_sents = split_sentences(summary)
    if len(summ_sents) >= 2:
        vec_red = TfidfVectorizer(lowercase=True, stop_words='english',
                                  ngram_range=ngram_range, max_df=max_df, min_df=min_df)
        X_red = vec_red.fit_transform(summ_sents)
        C = cosine_similarity(X_red)
        # exclude diagonal
        redundancy = float((C.sum() - np.trace(C)) / (C.shape[0]*C.shape[1] - C.shape[0]))
    else:
        redundancy = 0.0

    # Top keywords in the summary (TF-IDF weighted)
    vec_kw = TfidfVectorizer(lowercase=True, stop_words='english',
                             ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    X_kw = vec_kw.fit_transform([summary])
    vocab = np.array(vec_kw.get_feature_names_out())
    scores = X_kw.toarray()[0]
    top_idx = scores.argsort()[::-1][:10]
    top_keywords = [(vocab[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    return {
        "tfidf_cosine_similarity": global_sim,
        "coverage@{:.2f}".format(coverage_threshold): coverage,
        "redundancy_avg_pairwise": redundancy,
        "top_keywords": top_keywords,
        "notes": "Higher similarity & coverage are good; lower redundancy is better."
    }

# Run it
metrics = tfidf_eval(source_text, summary, ngram_range=(1,2), coverage_threshold=0.10)
print("\n--- TF-IDF EVAL ---")
for k, v in metrics.items():
    if k == "top_keywords":
        print(f"{k}: {[w for w,_ in v]}")
    else:
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Persist alongside your summary file
with open("outputs/tfidf_eval.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("Saved TF-IDF metrics to outputs/tfidf_eval.json")

