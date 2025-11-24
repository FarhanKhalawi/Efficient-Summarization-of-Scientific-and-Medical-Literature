import os
import re
import json
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
# NLTK: tokenization (for sentence splitting)
# -------------------------------
import nltk
from nltk.tokenize import sent_tokenize


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


_ensure_nltk_data()

print("CUDA available:", torch.cuda.is_available())

# -------------------------------
# Config
# -------------------------------

model_name = "Qwen/Qwen3-Reranker-8B"
#model_name = "Qwen/Qwen3-0.6B"
TEST_CSV = "data/MeDAL/pretrain_subset/test.csv"
HUMAN_CSV = "data/MeDAL/pretrain_subset/human_summaries_for_rouge.csv"

#OUT_DIR = "outputs(Qwen3-0.6B)_GPT"
OUT_DIR = "outputs(Qwen3-Reranker-8B)_GPT"
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
        score: float in [0, 1] (proportion of hallucinated statements)
        raw_text: raw model output (for debugging only)
    """
    judge_prompt = (
        "You will evaluate how hallucinated the SUMMARY is compared to the SOURCE.\n"
        "Definition: A hallucination is any factual claim in the SUMMARY that is not supported by the SOURCE.\n\n"
        "Task:\n"
        "- Read the SOURCE carefully.\n"
        "- Read the SUMMARY carefully.\n"
        "- Decide how hallucinated the SUMMARY is.\n\n"
        "Give ONE number between 0 and 1 ONLY, where:\n"
        "0.0 = No hallucination at all (perfectly faithful)\n"
        "0.1 = Very tiny hallucination\n"
        "0.15 = Slight hallucination\n"
        "0.2 = Small hallucination\n"
        "0.25 = Small–moderate hallucination\n"
        "0.3 = Noticeable hallucination\n"
        "0.35 = Noticeable–moderate hallucination\n"
        "0.4 = Moderate hallucination\n"
        "0.45 = Moderate–strong hallucination\n"
        "0.5 = Half correct / half hallucinated\n"
        "0.55 = Strong-ish hallucination\n"
        "0.6 = Strong hallucination\n"
        "0.65 = Strong–very strong hallucination\n"
        "0.7 = Very strong hallucination\n"
        "0.8 = Severe hallucination\n"
        "0.9 = Extremely strong hallucination\n"
        "1.0 = Completely hallucinated\n\n"
        "IMPORTANT RULES:\n"
        "- Output ONLY the number.\n"
        "- No explanation.\n"
        "- No JSON.\n"
        "- Do NOT include any explanation, comments, or extra text.\n\n"
        f"SOURCE:\n{source_text}\n\n"
        f"SUMMARY (model):\n{model_summary}\n\n"
    )

    if human_summary is not None:
        judge_prompt += f"HUMAN reference summary (for context):\n{human_summary}\n\n"

    judge_prompt += "Now output only the JSON object with the hallucination_score."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        # If quota error / invalid key etc., return NaN so script continues
        return np.nan, f"OpenAI error: {e}"

    text_out = response.choices[0].message.content.strip()

    # Try to parse JSON
    try:
        obj = json.loads(text_out)
        score = float(obj.get("hallucination_score", 0.0))
        # Clip to [0, 1] just in case
        score = max(0.0, min(1.0, score))
        return score, text_out
    except Exception:
        # Fallback: try to extract a float if JSON is slightly malformed
        m = re.search(r"(\d*\.?\d+)", text_out)
        if not m:
            return np.nan, text_out
        score = float(m.group(1))
        score = max(0.0, min(1.0, score))
        return score, text_out


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

    # strip any <think>...</think> blocks if present
    generated_summary = re.sub(r"(?is)<think>.*?</think>\s*", "", generated_summary)
    generated_summary = re.sub(r"(?is)^.*?</think>\s*", "", generated_summary)

    # keep max 3 sentences
    sents = sent_tokenize(generated_summary)
    if len(sents) > 3:
        generated_summary = " ".join(sents[:3])

    print("\n--- SUMMARY (MODEL) ---\n")
    print(generated_summary)

    # TF-IDF similarity
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

    # ROUGE (model vs human)
    scores = scorer.score(human_summary, generated_summary)
    r1 = scores["rouge1"].fmeasure
    r2 = scores["rouge2"].fmeasure
    rL = scores["rougeL"].fmeasure
    print("\n--- ROUGE (model vs human) ---")
    print(f"ROUGE-1 F1: {r1:.4f}")
    print(f"ROUGE-2 F1: {r2:.4f}")
    print(f"ROUGE-L F1: {rL:.4f}")

    # ---------------------------
    # Hallucination: GPT-4o-mini (LLM-as-Judge)
    # ---------------------------
    print("\n--- GPT-4o-mini HALLUCINATION (LLM-as-Judge) ---")
    hallucination_gpt4, raw_gpt4 = gpt4_hallucination_judge(
        original_text,
        generated_summary,
        human_summary,
    )

    print(f"Hallucination score (GPT-4o-mini): {hallucination_gpt4:.4f}")

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

        # GPT-4o-mini numeric hallucination metric in [0, 1]
        "hallucination_gpt4": float(hallucination_gpt4) if not np.isnan(hallucination_gpt4) else np.nan,

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
ax.set_title("Correlation heatmap: TF-IDF & ROUGE")

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

# 2) Line plot: GPT-4o-mini hallucination (simple line only)
fig2, ax2 = plt.subplots(figsize=(6, 4))
x = df_all["sample_idx"].values

# Simple line: no markers, no legend, no grid
ax2.plot(x, df_all["hallucination_gpt4"].values)

ax2.set_xlabel("Sample index")
ax2.set_ylabel("Hallucination score")
ax2.set_title("GPT-4o-mini hallucination across samples")
ax2.set_ylim(0.0, 1.0)

fig2.tight_layout()
fig2.savefig(HALL_LINE_PNG, dpi=200)
plt.close(fig2)
print(f"Saved hallucination line plot: {HALL_LINE_PNG}")
