import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# -------------------------------
# NLTK: robust tokenization + lemmatization
# -------------------------------
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def _ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    for corpus in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)
    # POS tagger: try new + legacy names
    def _ensure_tagger():
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng"); return
        except LookupError:
            pass
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger"); return
        except LookupError:
            pass
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.data.find("taggers/averaged_perceptron_tagger_eng"); return
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.data.find("taggers/averaged_perceptron_tagger")
    try:
        _ensure_tagger()
    except LookupError:
        pass

_ensure_nltk_data()

def safe_pos_tag(tokens):
    try:
        return pos_tag(tokens, lang="eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            return pos_tag(tokens, lang="eng")
        except Exception:
            pass
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            return pos_tag(tokens, lang="eng")
        except Exception:
            return [(t, "NN") for t in tokens]

print("CUDA available:", torch.cuda.is_available())

# -------------------------------
# Config
# -------------------------------
model_name = "Qwen/Qwen3-0.6B"
TEST_CSV = "data/MeDAL/pretrain_subset/test.csv"
HUMAN_CSV = "data/MeDAL/pretrain_subset/human_summaries_for_rouge.csv"

# >>> Output directory + file paths <<<
OUT_DIR = "outputs(Qwen3-0.6B)"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV  = os.path.join(OUT_DIR, "results.csv")
HEATMAP_PNG = os.path.join(OUT_DIR, "metric_correlation_heatmap.png")
BAR_PNG     = os.path.join(OUT_DIR, "avg_metrics_bar.png")

# How many items to process
NUM_ITEMS = 90

# -------------------------------
# Load data and model
# -------------------------------
test_df = pd.read_csv(TEST_CSV)
human_df = pd.read_csv(HUMAN_CSV)

print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!")

# Explicit GenerationConfig (silences modified-default notices)
base = model.generation_config
gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=20,
    max_new_tokens=256,
    min_new_tokens=40,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=base.bos_token_id,
    eos_token_id=base.eos_token_id,
)

# -------------------------------
# Helpers for hallucination check
# -------------------------------
SCIENTIFIC_FILLERS = {
    "study","studies","analysis","analyses","model","group","groups","control",
    "function","functions","results","findings","indicating","showed","observed",
    "parameters","measurements","disease","method","methods","purpose","aim"
}
wnl = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def _to_wn_pos(tag: str):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def normalize_and_lemmatize(text: str):
    toks = word_tokenize(text)
    tagged = safe_pos_tag(toks)
    lemmas = []
    for tok, tag in tagged:
        t = tok.lower()
        if t.isalpha() and t not in STOPWORDS:
            lemmas.append(wnl.lemmatize(t, pos=_to_wn_pos(tag)))
    return lemmas

# -------------------------------
# Summarize + score + log CSV
# -------------------------------
rows_out = []
run_stamp = datetime.now().isoformat(timespec="seconds")

n_items = min(NUM_ITEMS, len(test_df))
print(f"Processing {n_items} items ...")

for i in range(n_items):
    print(f"\n====================== SAMPLE {i+1} ======================\n")
    row = test_df.iloc[i]
    original_text = str(row["TEXT"])
    abstract_id = int(row["ABSTRACT_ID"])

    ref_row = human_df[human_df["ABSTRACT_ID"] == abstract_id]
    if ref_row.empty:
        print(f"No human summary for ABSTRACT_ID={abstract_id}")
        continue
    human_summary = ref_row["HUMAN_SUMMARY"].iloc[0]

    prompt = (
        "Summarize the following scientific text in EXACTLY 2 or 3 sentences. "
        "ONLY use facts explicitly present in the text. "
        "If a number or unit is not present in the text, do NOT invent it. "
        "Focus strictly on purpose, methods, and main findings.\n\n"
        f"{original_text}\n\nSummary:"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            generation_config=gen_cfg
        )

    # strip prompt
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    generated_summary = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    # strip any <think>...</think>
    generated_summary = re.sub(r"(?is)<think>.*?</think>\s*", "", generated_summary)
    generated_summary = re.sub(r"(?is)^.*?</think>\s*", "", generated_summary)
    # cap at 3 sentences
    sents = sent_tokenize(generated_summary)
    if len(sents) > 3:
        generated_summary = " ".join(sents[:3])

    print("\n--- SUMMARY (MODEL) ---\n")
    print(generated_summary)

    # TF-IDF
    corpus = [original_text, generated_summary]
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    print("\n--- TF-IDF SIMILARITY ---")
    print(f"Cosine similarity (original vs summary): {similarity:.4f}")

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(human_summary, generated_summary)
    r1 = scores['rouge1'].fmeasure
    r2 = scores['rouge2'].fmeasure
    rL = scores['rougeL'].fmeasure
    print("\n--- ROUGE (model vs human) ---")
    print(f"ROUGE-1 F1: {r1:.4f}")
    print(f"ROUGE-2 F1: {r2:.4f}")
    print(f"ROUGE-L F1: {rL:.4f}")

    # -------------------------------------------------------
    # Hallucination check (method explained)
    # -------------------------------------------------------
    # Idea:
    # 1) Normalize & lemmatize tokens from source text, model summary, and human summary.
    # 2) Mark a summary token as "unsupported" if it is not present in the source token set
    #    and not a common scientific filler (e.g., "study", "method", etc.).
    # 3) "Truly unsupported" = unsupported AND not present in the human summary either.
    # 4) Separately detect numbers in the summary that do not appear in the source text.
    # 5) Compute unsupported_ratio = (# truly unsupported tokens) / (summary token count).
    # 6) Mix with a softening factor based on ROUGE-1 (higher ROUGE -> slightly lower penalty)
    #    and add a small penalty (0.05) per hallucinated number.
    # 7) Map the final score to LOW / MEDIUM / HIGH levels.
    print("\n--- HALLUCINATION CHECK ---")
    src_tokens = normalize_and_lemmatize(original_text)
    sum_tokens = normalize_and_lemmatize(generated_summary)
    ref_tokens = normalize_and_lemmatize(human_summary)

    src_set, ref_set = set(src_tokens), set(ref_tokens)
    unsupported_src = [w for w in sum_tokens if w not in src_set and w not in SCIENTIFIC_FILLERS]
    truly_unsupported = [w for w in unsupported_src if w not in ref_set]

    src_numbers = set(re.findall(r"\d+(?:\.\d+)?", original_text.lower()))
    sum_numbers = set(re.findall(r"\d+(?:\.\d+)?", generated_summary.lower()))
    numeric_hallucinated = [n for n in sum_numbers if n not in src_numbers]

    unsupported_ratio = (len(truly_unsupported) / len(sum_tokens)) if sum_tokens else 0.0
    soften = 1.0 - min(r1, 0.6) * 0.3
    hallucination_score = min(1.0, unsupported_ratio * soften + 0.05 * len(numeric_hallucinated))

    if hallucination_score < 0.15:
        level = "LOW"
    elif hallucination_score < 0.35:
        level = "MEDIUM"
    else:
        level = "HIGH"

    print(f"Unsupported tokens: {truly_unsupported}")
    print(f"Numeric hallucinations: {numeric_hallucinated}")
    print(f"Hallucination score: {hallucination_score:.3f} -> {level}")

    rows_out.append({
        "run_timestamp": run_stamp,
        "sample_idx": i,
        "ABSTRACT_ID": abstract_id,
        "orig_len": len(original_text),
        "sum_len": len(generated_summary),
        "tfidf_cosine": float(similarity),
        "rouge1_f1": float(r1),
        "rouge2_f1": float(r2),
        "rougeL_f1": float(rL),
        "hallucination_score": float(hallucination_score),
        "hallucination_level": level,
        "unsupported_count": len(truly_unsupported),
        "numeric_hallucinated_count": len(numeric_hallucinated),
        "unsupported_tokens": " ".join(truly_unsupported),
        "numeric_hallucinated": " ".join(numeric_hallucinated),
        "generated_summary": generated_summary
    })

# Save CSV (append if exists)
out_df = pd.DataFrame(rows_out)
if os.path.exists(OUT_CSV):
    out_df.to_csv(OUT_CSV, mode="a", header=False, index=False, encoding="utf-8")
else:
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"\nSaved results to: {OUT_CSV}")

# -------------------------------
# Visualizations
# -------------------------------
metrics_cols = ["tfidf_cosine", "rouge1_f1", "rouge2_f1", "rougeL_f1", "hallucination_score"]

# Load full CSV (in case we appended)
df_all = pd.read_csv(OUT_CSV)

# 1) Correlation heatmap
corr = df_all[metrics_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, aspect="auto")  # default colormap & single plot
ax.set_xticks(np.arange(len(metrics_cols)))
ax.set_yticks(np.arange(len(metrics_cols)))
ax.set_xticklabels(metrics_cols, rotation=45, ha="right")
ax.set_yticklabels(metrics_cols)
ax.set_title("Metric correlation heatmap")

# annotate cells
for irow in range(corr.shape[0]):
    for jcol in range(corr.shape[1]):
        ax.text(jcol, irow, f"{corr.values[irow, jcol]:.2f}", ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(HEATMAP_PNG, dpi=200)
plt.close(fig)
print(f"Saved heatmap: {HEATMAP_PNG}")

# 2) Average metrics bar chart
means = df_all[metrics_cols].mean()
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(means.index, means.values)  # default colors; single plot
ax2.set_title("Average metrics over 90 items")
ax2.set_ylabel("Score")
ax2.tick_params(axis='x', labelrotation=20)
fig2.tight_layout()
fig2.savefig(BAR_PNG, dpi=200)
plt.close(fig2)
print(f"Saved bar chart: {BAR_PNG}")
