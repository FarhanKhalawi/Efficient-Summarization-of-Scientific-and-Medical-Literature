import re
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer  

print("CUDA available:", torch.cuda.is_available())

model_name = "Qwen/Qwen3-0.6B"
TEST_CSV = "data/MeDAL/pretrain_subset/test.csv"
HUMAN_CSV = "data/MeDAL/pretrain_subset/human_summaries_for_rouge.csv"

# -------------------------------------------------------------------------
# 1) Load data
# -------------------------------------------------------------------------
test_df = pd.read_csv(TEST_CSV)
human_df = pd.read_csv(HUMAN_CSV)

row = test_df.iloc[0]
original_text = str(row["TEXT"])
abstract_id = int(row["ABSTRACT_ID"])

# match human summary by ABSTRACT_ID
ref_row = human_df[human_df["ABSTRACT_ID"] == abstract_id]
if ref_row.empty:
    raise ValueError(f"No human summary found for ABSTRACT_ID={abstract_id}")
human_summary = ref_row["HUMAN_SUMMARY"].iloc[0]

print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!")

# -------------------------------------------------------------------------
# 2) Summarization
# -------------------------------------------------------------------------
prompt = (
    "Summarize the following scientific text in exactly 2 or 3 concise sentences, "
    "focusing only on the study’s purpose, methods, and main findings, "
    "keeping important medical/experimental details:\n\n"
    f"{original_text}\n\nSummary:"
)

messages = [{"role": "user", "content": prompt}]

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

generated_summary = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n--- SUMMARY (MODEL) ---\n")
print(generated_summary)

# -------------------------------------------------------------------------
# 3) TF-IDF evaluation
# -------------------------------------------------------------------------
corpus = [original_text, generated_summary]

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2)
)
tfidf_matrix = vectorizer.fit_transform(corpus)
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print("\n--- TF-IDF SIMILARITY ---")
print(f"Cosine similarity (original vs summary): {similarity:.4f}")

# -------------------------------------------------------------------------
# 4) ROUGE evaluation
# -------------------------------------------------------------------------
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)
scores = scorer.score(human_summary, generated_summary)

print("\n--- ROUGE (model vs human) ---")
print(f"ROUGE-1  F1: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2  F1: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L  F1: {scores['rougeL'].fmeasure:.4f}")

# -------------------------------------------------------------------------
# 5) Hallucination check (source-aware + human-aware)
# -------------------------------------------------------------------------
print("\n--- HALLUCINATION (source + human) ---")

def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

src_norm = normalize(original_text)
sum_norm = normalize(generated_summary)
ref_norm = normalize(human_summary)

stopwords = {
    "the","a","an","of","to","and","in","for","on","at","by","with",
    "this","that","these","those","was","were","is","are","be","been",
    "being","as","it","its","they","their","we","our","or","from",
    "into","than","which","such","using"
}

scientific_fillers = {
    "study","studies","analysis","analyses","model","compared","compare",
    "sheep","group","groups","reactive","nonreactive","control","pulmonary",
    "bal","bronchoalveolar","lavage","disease","airway","ascaris","antigen",
    "results","finding","findings","indicating","indicated","showed","observed",
    "parameters","tests","function","functions","measurement","measurements"
}

def simple_lemma(w: str) -> str:
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w

src_tokens = [simple_lemma(w) for w in re.findall(r"[a-zA-Z]+", src_norm) if w not in stopwords]
sum_tokens = [simple_lemma(w) for w in re.findall(r"[a-zA-Z]+", sum_norm) if w not in stopwords]
ref_tokens = [simple_lemma(w) for w in re.findall(r"[a-zA-Z]+", ref_norm) if w not in stopwords]

src_set = set(src_tokens)
ref_set = set(ref_tokens)

# 1) unsupported vs source
unsupported_src = [
    w for w in sum_tokens
    if w not in src_set and w not in scientific_fillers
]

# 2) but if it's in human ref, we forgive it
truly_unsupported = [w for w in unsupported_src if w not in ref_set]

# 3) numeric hallucinations
src_numbers = set(re.findall(r"\d+(?:\.\d+)?", src_norm))
sum_numbers = set(re.findall(r"\d+(?:\.\d+)?", sum_norm))
numeric_hallucinated = [n for n in sum_numbers if n not in src_numbers]

if len(sum_tokens) > 0:
    unsupported_ratio = len(truly_unsupported) / len(sum_tokens)
else:
    unsupported_ratio = 0.0

# soften by ROUGE-1 
rouge1_f1 = scores["rouge1"].fmeasure
soften = 1.0 - min(rouge1_f1, 0.6) * 0.3  
hallucination_score = min(1.0, unsupported_ratio * soften + 0.05 * len(numeric_hallucinated))

if hallucination_score < 0.15:
    level = "LOW"
elif hallucination_score < 0.35:
    level = "MEDIUM"
else:
    level = "HIGH"

print(f"Unsupported tokens (after human check): {truly_unsupported}")
print(f"Numeric hallucinations: {numeric_hallucinated}")
print(f"Hallucination score: {hallucination_score:.3f} -> {level}")

