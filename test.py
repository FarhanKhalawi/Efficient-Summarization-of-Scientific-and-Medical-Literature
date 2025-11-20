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
    GenerationConfig,
    AutoModelForSequenceClassification,
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

    # POS tagger: new + legacy names
    def _ensure_tagger():
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            return
        except LookupError:
            pass
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
            return
        except LookupError:
            pass
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            return
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.data.find("taggers/averaged_perceptron_tagger")

    try:
        _ensure_tagger()
    except LookupError:
        # fall back: handled in safe_pos_tag
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

OUT_DIR = "outputs(Qwen3-0.6B)_switchable"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "results.csv")
HEATMAP_PNG = os.path.join(OUT_DIR, "metric_correlation_heatmap.png")
BAR_PNG = os.path.join(OUT_DIR, "avg_metrics_bar.png")
SCATTER_PNG = os.path.join(OUT_DIR, "nli_vs_hallucination_scatter.png")

N_SAMPLES = 90  # run over 90 items
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Choose hallucination method: "heuristic" or "nli"
HALLUCINATION_METHOD = "nli"  # change to "heuristic" if you want the lightweight method

# -------------------------------
# Load data and model
# -------------------------------
test_df = pd.read_csv(TEST_CSV)
human_df = pd.read_csv(HUMAN_CSV)

print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
print("Model loaded successfully!")

base = model.generation_config
gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=20,
    max_new_tokens=256,
    min_new_tokens=40,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=getattr(base, "bos_token_id", None),
    eos_token_id=getattr(base, "eos_token_id", tokenizer.eos_token_id),
)

# -------------------------------
# Heuristic hallucination helpers
# -------------------------------
SCIENTIFIC_FILLERS = {
    "study",
    "studies",
    "analysis",
    "analyses",
    "model",
    "group",
    "groups",
    "control",
    "function",
    "functions",
    "results",
    "findings",
    "indicating",
    "showed",
    "observed",
    "parameters",
    "measurements",
    "disease",
    "method",
    "methods",
    "purpose",
    "aim",
}
wnl = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))


def _to_wn_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
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


def heuristic_hallucination_score(
    source_text: str, summary_text: str, rouge1_f1: float
) -> (float, str, dict):
    src_tokens = normalize_and_lemmatize(source_text)
    sum_tokens = normalize_and_lemmatize(summary_text)

    src_set = set(src_tokens)
    unsupported_src = [
        w for w in sum_tokens if w not in src_set and w not in SCIENTIFIC_FILLERS
    ]

    src_numbers = set(re.findall(r"\d+(?:\.\d+)?", source_text.lower()))
    sum_numbers = set(re.findall(r"\d+(?:\.\d+)?", summary_text.lower()))
    numeric_hallucinated = [n for n in sum_numbers if n not in src_numbers]

    unsupported_ratio = (len(unsupported_src) / len(sum_tokens)) if sum_tokens else 0.0
    soften = 1.0 - min(rouge1_f1, 0.6) * 0.3
    score = min(1.0, unsupported_ratio * soften + 0.05 * len(numeric_hallucinated))

    if score < 0.15:
        level = "LOW"
    elif score < 0.35:
        level = "MEDIUM"
    else:
        level = "HIGH"

    dbg = {
        "unsupported_tokens": unsupported_src,
        "numeric_hallucinated": numeric_hallucinated,
    }
    return score, level, dbg


# -------------------------------
# NLI-based factuality helpers
# -------------------------------
ENTAILMENT_LABEL = 2  # roberta-large-mnli label order: [contradiction, neutral, entailment]
if HALLUCINATION_METHOD == "nli":
    nli_name = "roberta-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(
        model.device
    )
    nli_model.eval()


def _chunk_text_by_tokens(text, max_tokens=420):
    sents = sent_tokenize(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        l = len(s.split())
        if cur_len + l > max_tokens and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], l
        else:
            cur.append(s)
            cur_len += l
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


@torch.no_grad()
def nli_entailment_score(source_text: str, summary_text: str) -> float:
    if not summary_text.strip():
        return 0.0
    sum_sents = [s.strip() for s in sent_tokenize(summary_text) if s.strip()]
    if not sum_sents:
        sum_sents = [summary_text.strip()]
    src_chunks = _chunk_text_by_tokens(source_text, max_tokens=420)

    entail_scores = []
    for hypo in sum_sents:
        probs_for_hypo = []
        for prem in src_chunks:
            enc = nli_tok(
                prem,
                hypo,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(nli_model.device)
            logits = nli_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            probs_for_hypo.append(probs[ENTAILMENT_LABEL])
        entail_scores.append(float(np.max(probs_for_hypo)))
    return float(np.mean(entail_scores)) if entail_scores else 0.0


def nli_score_to_level(h: float) -> str:
    # h here is hallucination_score (= 1 - entailment)
    if h < 0.15:
        return "LOW"
    elif h < 0.35:
        return "MEDIUM"
    return "HIGH"


# -------------------------------
# Summarize + score + log CSV
# -------------------------------
rows_out = []
run_stamp = datetime.now().isoformat(timespec="seconds")

limit = min(N_SAMPLES, len(test_df))
for i in range(limit):
    print(f"\n====================== SAMPLE {i+1} / {limit} ======================\n")
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
        enable_thinking=False,
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            generation_config=gen_cfg,
        )

    # strip prompt
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    generated_summary = tokenizer.decode(
        output_ids, skip_special_tokens=True
    ).strip()

    # strip any <think>...</think>
    generated_summary = re.sub(r"(?is)<think>.*?</think>\s*", "", generated_summary)
    generated_summary = re.sub(r"(?is)^.*?</think>\s*", "", generated_summary)

    # cap at 3 sentences
    sents = sent_tokenize(generated_summary)
    if len(sents) > 3:
        generated_summary = " ".join(sents[:3])

    print("\n--- SUMMARY (MODEL) ---\n")
    print(generated_summary)

    # TF-IDF (source vs summary)
    corpus = [original_text, generated_summary]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    print("\n--- TF-IDF SIMILARITY ---")
    print(f"Cosine similarity (original vs summary): {similarity:.4f}")

    # ROUGE (human vs model summary)
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(human_summary, generated_summary)
    r1 = scores["rouge1"].fmeasure
    r2 = scores["rouge2"].fmeasure
    rL = scores["rougeL"].fmeasure
    print("\n--- ROUGE (model vs human) ---")
    print(f"ROUGE-1 F1: {r1:.4f}")
    print(f"ROUGE-2 F1: {r2:.4f}")
    print(f"ROUGE-L F1: {rL:.4f}")

    # Hallucination / factuality
    if HALLUCINATION_METHOD == "nli":
        print("\n--- NLI FACTUALITY (ENTAILMENT) ---")
        entail = nli_entailment_score(original_text, generated_summary)
        hallucination_score = 1.0 - entail  # <- mirror metric (expected strong -corr)
        level = nli_score_to_level(hallucination_score)
        print(f"Avg entailment: {entail:.3f}")
        print(f"Hallucination score: {hallucination_score:.3f} -> {level}")
        dbg_fields = {}
        row_out_extra = {
            "nli_avg_entailment": float(entail),
        }
    else:
        print("\n--- HEURISTIC HALLUCINATION ---")
        score, level, dbg = heuristic_hallucination_score(
            original_text, generated_summary, r1
        )
        hallucination_score = score
        print(f"Unsupported tokens: {dbg['unsupported_tokens']}")
        print(f"Numeric hallucinations: {dbg['numeric_hallucinated']}")
        print(f"Hallucination score: {hallucination_score:.3f} -> {level}")
        dbg_fields = {
            "unsupported_tokens": " ".join(dbg["unsupported_tokens"]),
            "numeric_hallucinated": " ".join(dbg["numeric_hallucinated"]),
        }
        row_out_extra = {}

    row_out = {
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
        "generated_summary": generated_summary,
    }
    row_out.update(row_out_extra)
    row_out.update(dbg_fields)
    rows_out.append(row_out)

# Save CSV (append if exists)
out_df = pd.DataFrame(rows_out)
if os.path.exists(OUT_CSV):
    out_df.to_csv(
        OUT_CSV,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )
else:
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"\nSaved results to: {OUT_CSV}")

# -------------------------------
# Visualizations + sanity checks
# -------------------------------
df_all = pd.read_csv(OUT_CSV)

# metrics for heatmap
if HALLUCINATION_METHOD == "nli":
    metrics_cols = [
        "tfidf_cosine",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "nli_avg_entailment",
        "hallucination_score",
    ]
else:
    metrics_cols = [
        "tfidf_cosine",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "hallucination_score",
    ]

# --- sanity checks if we are using NLI ---
if HALLUCINATION_METHOD == "nli" and "nli_avg_entailment" in df_all.columns:
    print("\nNLI & hallucination summary stats:")
    print(df_all[["nli_avg_entailment", "hallucination_score"]].describe())

    # value range checks (if using prob-like scores)
    if not df_all["hallucination_score"].between(0, 1).all():
        print("Warning: some hallucination_score values are outside [0, 1].")
    if not df_all["nli_avg_entailment"].between(0, 1).all():
        print("Warning: some nli_avg_entailment values are outside [0, 1].")

    print("\nCorrelation NLI vs hallucination:")
    print(df_all[["nli_avg_entailment", "hallucination_score"]].corr())

    # scatter: NLI vs hallucination
    fig_sc, ax_sc = plt.subplots(figsize=(5, 4))
    ax_sc.scatter(
        df_all["nli_avg_entailment"], df_all["hallucination_score"], alpha=0.6
    )
    ax_sc.set_xlabel("NLI average entailment")
    ax_sc.set_ylabel("Hallucination score (1 - entailment)")
    ax_sc.set_title("NLI vs hallucination")
    fig_sc.tight_layout()
    fig_sc.savefig(SCATTER_PNG, dpi=200)
    plt.close(fig_sc)
    print(f"Saved scatter plot: {SCATTER_PNG}")

# 1) Correlation heatmap
corr = df_all[metrics_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, aspect="auto")
ax.set_xticks(np.arange(len(metrics_cols)))
ax.set_yticks(np.arange(len(metrics_cols)))
ax.set_xticklabels(metrics_cols, rotation=45, ha="right")
ax.set_yticklabels(metrics_cols)
ax.set_title("Metric correlation heatmap")

for irow in range(corr.shape[0]):
    for jcol in range(corr.shape[1]):
        ax.text(
            jcol,
            irow,
            f"{corr.values[irow, jcol]:.2f}",
            ha="center",
            va="center",
        )

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(HEATMAP_PNG, dpi=200)
plt.close(fig)
print(f"Saved heatmap: {HEATMAP_PNG}")

# 2) Average metrics bar chart (using all rows currently in CSV)
means = df_all[metrics_cols].mean()
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(means.index, means.values)
ax2.set_title(f"Average metrics over {len(df_all)} items")
ax2.set_ylabel("Score")
ax2.tick_params(axis="x", labelrotation=20)
fig2.tight_layout()
fig2.savefig(BAR_PNG, dpi=200)
plt.close(fig2)
print(f"Saved bar chart: {BAR_PNG}")

