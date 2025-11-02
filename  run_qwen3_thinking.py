import re
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("CUDA available:", torch.cuda.is_available())

model_name = "Qwen/Qwen3-0.6B"
CSV_PATH = "data/MeDAL/pretrain_subset/test.csv"

# -------------------------------------------------------------------------
# Data load
# -------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
original_text = df["TEXT"].astype(str).iloc[0]

print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!")

# -------------------------------------------------------------------------
# Summarization
# -------------------------------------------------------------------------
prompt = (
    "Summarize the following scientific text in exactly 2 or 3 concise sentences, "
    "focusing only on the study’s purpose, methods, and main findings. "
    "Avoid listing specific numerical values:\n\n"
    f"{original_text}\n\nSummary:"
)

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

try:
    # 151668 is </think>
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n--- SUMMARY ---\n")
print(content)

# -------------------------------------------------------------------------
# TF-IDF evaluation
# -------------------------------------------------------------------------
corpus = [original_text, content]

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)
tfidf_matrix = vectorizer.fit_transform(corpus)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print("\n--- TF-IDF SIMILARITY ---")
print(f"Cosine similarity (original vs summary): {similarity:.4f}")

# =============================================================================
# Improved Hallucination Evaluation
# =============================================================================
print("\n--- HALLUCINATION EVALUATION ---")

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

src_norm = normalize_text(original_text)
sum_norm = normalize_text(content)

# words we will NOT treat as hallucinations even if not in source
scientific_fillers = {
    "study", "studies", "paper", "work", "analysis", "analyses",
    "showed", "demonstrated", "reported", "compared", "assessment",
    "groups", "group", "reactive", "nonreactive", "control", "model",
    "parameters", "findings", "results", "disease", "airway"
}

stopwords = {
    "the","a","an","of","to","and","in","for","on","at","by","with",
    "this","that","these","those","was","were","is","are","be","been",
    "being","as","it","its","they","their","we","our","or","from",
    "into","than","which","such","using"
}

def simple_lemma(w: str) -> str:
    # super light lemmatization
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w

src_tokens = [simple_lemma(w) for w in re.findall(r"[a-zA-Z]+", src_norm)
              if w not in stopwords]
sum_tokens = [simple_lemma(w) for w in re.findall(r"[a-zA-Z]+", sum_norm)
              if w not in stopwords]

src_set = set(src_tokens)

unsupported = [
    w for w in sum_tokens
    if w not in src_set and w not in scientific_fillers
]

# numeric hallucination check
src_numbers = set(re.findall(r"\d+(?:\.\d+)?", original_text))
sum_numbers = set(re.findall(r"\d+(?:\.\d+)?", content))
numeric_hallucinated = [n for n in sum_numbers if n not in src_numbers]

# base ratio
unsupported_ratio = len(unsupported) / len(sum_tokens) if len(sum_tokens) else 0.0

# adjust by tf-idf: if summary is well aligned lexically, we soften the penalty
if similarity > 0.35:
    unsupported_ratio *= 0.7

hallucination_score = min(1.0, unsupported_ratio + 0.05 * len(numeric_hallucinated))

if hallucination_score < 0.15:
    level = "LOW"
elif hallucination_score < 0.35:
    level = "MEDIUM"
else:
    level = "HIGH"

print(f"Unsupported tokens (not in source): {unsupported}")
print(f"Numeric hallucinations: {numeric_hallucinated}")
print(f"Hallucination score: {hallucination_score:.3f} -> {level}")
