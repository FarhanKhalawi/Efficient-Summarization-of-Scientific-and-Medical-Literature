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
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# -------------------------------
# OpenAI: GPT-4o-mini as LLM-as-Judge
# -------------------------------
import openai as openai_module
from openai import OpenAI

# client will read OPENAI_API_KEY from environment
client = OpenAI()

# -------------------------------
# NLTK: tokenization + lemmatization
# -------------------------------
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def _ensure_nltk_data():
    # tokenizers
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    # corpora
    for corpus in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)

    # POS tagger
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

OUT_DIR = "outputs(Qwen3-0.6B)"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "results.csv")
HEATMAP_PNG = os.path.join(OUT_DIR, "metric_correlation_heatmap.png")
HALL_LINE_PNG = os.path.join(OUT_DIR, "hallucination_line.png")

N_SAMPLES = 90
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------------
# Load data
# -------------------------------
test_df = pd.read_csv(TEST_CSV)
human_df = pd.read_csv(HUMAN_CSV)

# -------------------------------
# Load summarization model
# -------------------------------
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
    # you can keep sampling, or switch to greedy by do_sample=False
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
# GPT-4o-mini hallucination metric (LLM-as-Judge)
# -------------------------------
def gpt4_hallucination_judge(
    source_text: str,
    model_summary: str,
    human_summary: str | None = None,
):
    """
    Uses GPT-4o-mini as hallucination detector.
    Returns:
        hallucinated: True / False / None
        explanation: str
    """
    judge_prompt = (
        "You are an expert evaluator of scientific summaries.\n\n"
        "You will be given:\n"
        "1) SOURCE: the original scientific text.\n"
        "2) SUMMARY: a model-generated summary.\n"
        "3) Optionally a HUMAN reference summary.\n\n"
        "Task:\n"
        "- Decide if the SUMMARY contains any hallucinations: facts, numbers, "
        "or claims that are NOT supported by the SOURCE text.\n"
        "- Ignore style and grammar. Focus only on factual faithfulness.\n\n"
        "Output format (IMPORTANT):\n"
        "First line MUST be exactly one of:\n"
        "  hallucinated: YES\n"
        "  hallucinated: NO\n"
        "Then, on the following lines, briefly explain your reasoning.\n\n"
        f"SOURCE:\n{source_text}\n\n"
        f"SUMMARY (model):\n{model_summary}\n\n"
    )

    if human_summary is not None:
        judge_prompt += f"HUMAN reference summary (for context):\n{human_summary}\n\n"

    judge_prompt += "Now give your decision following the required format."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4o" if you prefer
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful and conservative hallucination detector.",
                },
                {
                    "role": "user",
                    "content": judge_prompt,
                },
            ],
            temperature=0,
        )
    except openai_module.OpenAIError as e:
        # If quota error / invalid key etc., return None so script continues
        return None, f"OpenAI error: {e}"

    text_out = response.choices[0].message.content.strip()
    lines = [l for l in text_out.splitlines() if l.strip()]
    if not lines:
        return None, text_out

    first_line = lines[0].lower()
    explanation = "\n".join(lines[1:]).strip()

    if "hallucinated: yes" in first_line:
        return True, explanation or text_out
    if "hallucinated: no" in first_line:
        return False, explanation or text_out

    # fallback if format not followed
    return None, text_out


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


# -------------------------------
# Main loop: summarise + score
# -------------------------------
rows_out = []
run_stamp = datetime.now().isoformat(timespec="seconds")

limit = min(N_SAMPLES, len(test_df))
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)

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

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    generated_summary = tokenizer.decode(
        output_ids, skip_special_tokens=True
    ).strip()

    generated_summary = re.sub(r"(?is)<think>.*?</think>\s*", "", generated_summary)
    generated_summary = re.sub(r"(?is)^.*?</think>\s*", "", generated_summary)

    sents = sent_tokenize(generated_summary)
    if len(sents) > 3:
        generated_summary = " ".join(sents[:3])

    print("\n--- SUMMARY (MODEL) ---\n")
    print(generated_summary)

    # TF-IDF
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

    # ROUGE
    scores = scorer.score(human_summary, generated_summary)
    r1 = scores["rouge1"].fmeasure
    r2 = scores["rouge2"].fmeasure
    rL = scores["rougeL"].fmeasure
    print("\n--- ROUGE (model vs human) ---")
    print(f"ROUGE-1 F1: {r1:.4f}")
    print(f"ROUGE-2 F1: {r2:.4f}")
    print(f"ROUGE-L F1: {rL:.4f}")

    # ---------------------------
    # Hallucination way 1: GPT-4o-mini (LLM-as-Judge)
    # ---------------------------
    print("\n--- GPT-4o-mini HALLUCINATION (LLM-as-Judge) ---")
    hallucinated_gpt4, expl_gpt4 = gpt4_hallucination_judge(
        original_text,
        generated_summary,
        human_summary,
    )
    if hallucinated_gpt4 is True:
        hallucination_gpt4 = 1.0
        level_gpt4 = "HIGH"
    elif hallucinated_gpt4 is False:
        hallucination_gpt4 = 0.0
        level_gpt4 = "LOW"
    else:
        hallucination_gpt4 = np.nan
        level_gpt4 = "UNKNOWN"

    print(f"Hallucinated (GPT-4o-mini): {hallucinated_gpt4}")
    print("Explanation:")
    print(expl_gpt4)

    # ---------------------------
    # Hallucination way 2: heuristic
    # ---------------------------
    print("\n--- HEURISTIC HALLUCINATION (TRULY UNSUPPORTED) ---")
    src_tokens = normalize_and_lemmatize(original_text)
    sum_tokens = normalize_and_lemmatize(generated_summary)
    ref_tokens = normalize_and_lemmatize(human_summary)

    src_set, ref_set = set(src_tokens), set(ref_tokens)
    unsupported_src = [
        w for w in sum_tokens
        if w not in src_set and w not in SCIENTIFIC_FILLERS
    ]
    truly_unsupported = [w for w in unsupported_src if w not in ref_set]

    src_numbers = set(re.findall(r"\d+(?:\.\d+)?", original_text.lower()))
    sum_numbers = set(re.findall(r"\d+(?:\.\d+)?", generated_summary.lower()))
    numeric_hallucinated = [n for n in sum_numbers if n not in src_numbers]

    unsupported_ratio = (len(truly_unsupported) / len(sum_tokens)) if sum_tokens else 0.0
    soften = 1.0 - min(r1, 0.6) * 0.3
    hallucination_heur = min(
        1.0,
        unsupported_ratio * soften + 0.05 * len(numeric_hallucinated),
    )

    if hallucination_heur < 0.15:
        level_heur = "LOW"
    elif hallucination_heur < 0.35:
        level_heur = "MEDIUM"
    else:
        level_heur = "HIGH"

    print(f"Unsupported tokens (truly unsupported): {truly_unsupported}")
    print(f"Numeric hallucinations: {numeric_hallucinated}")
    print(f"Heuristic hallucination score: {hallucination_heur:.3f} -> {level_heur}")

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

        # GPT-4o-mini metric
        "hallucination_gpt4": float(hallucination_gpt4) if not np.isnan(hallucination_gpt4) else np.nan,
        "hallucination_level_gpt4": level_gpt4,
        "gpt4_explanation": expl_gpt4,

        # heuristic metric
        "hallucination_heuristic": float(hallucination_heur),
        "hallucination_level_heuristic": level_heur,
        "unsupported_count": len(truly_unsupported),
        "numeric_hallucinated_count": len(numeric_hallucinated),
        "unsupported_tokens": " ".join(truly_unsupported),
        "numeric_hallucinated": " ".join(numeric_hallucinated),
        "generated_summary": generated_summary,
    })

# Save CSV
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
# Visualisations
# -------------------------------
df_all = pd.read_csv(OUT_CSV)

# 1) Heatmap: ONLY TF-IDF + ROUGE
metrics_cols_heatmap = [
    "tfidf_cosine",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
]
corr = df_all[metrics_cols_heatmap].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, aspect="auto")
ax.set_xticks(np.arange(len(metrics_cols_heatmap)))
ax.set_yticks(np.arange(len(metrics_cols_heatmap)))
ax.set_xticklabels(metrics_cols_heatmap, rotation=45, ha="right")
ax.set_yticklabels(metrics_cols_heatmap)
ax.set_title("Correlation heatmap: TF-IDF & ROUGE only")

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

# 2) Line plot: GPT-4o-mini vs heuristic hallucination
fig2, ax2 = plt.subplots(figsize=(6, 4))
x = df_all["sample_idx"].values

ax2.plot(
    x,
    df_all["hallucination_gpt4"].values,
    marker="o",
    label="GPT-4o-mini hallucination",
)
ax2.plot(
    x,
    df_all["hallucination_heuristic"].values,
    marker="s",
    label="Heuristic hallucination",
)

ax2.set_xlabel("Sample index")
ax2.set_ylabel("Hallucination score (0/1 or 0–1)")
ax2.set_title("GPT-4o-mini vs heuristic hallucination across samples")
ax2.set_ylim(0.0, 1.0)
ax2.grid(True)
ax2.legend()
fig2.tight_layout()
fig2.savefig(HALL_LINE_PNG, dpi=200)
plt.close(fig2)
print(f"Saved hallucination line plot: {HALL_LINE_PNG}")
