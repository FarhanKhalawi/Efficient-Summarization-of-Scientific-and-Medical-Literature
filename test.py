
import re
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# -------------------------------
# NLTK: robust tokenization + lemmatization
# -------------------------------
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def _ensure_nltk_data():
    """Download needed NLTK resources if missing."""
    # Punkt
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        # Some envs also need this; ignore if absent
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    # Stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    # WordNet (+ multilingual mapping)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    # POS tagger: check both the new and the legacy resource names
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
        # Try new name first, then legacy
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            return
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            # last attempt; may still fail if offline
            nltk.data.find("taggers/averaged_perceptron_tagger")

    try:
        _ensure_tagger()
    except LookupError:
        # If still missing (e.g., offline), we’ll gracefully fallback later
        pass

_ensure_nltk_data()

def safe_pos_tag(tokens):
    """
    Try POS-tagging with explicit English model; if resources are missing,
    attempt downloads; if everything fails (e.g., offline), fall back to nouns.
    """
    try:
        return pos_tag(tokens, lang="eng")
    except LookupError:
        # Try downloading the new resource name
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            return pos_tag(tokens, lang="eng")
        except Exception:
            pass
        # Try the legacy one
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            return pos_tag(tokens, lang="eng")
        except Exception:
            # Fallback: tag all as nouns
            return [(t, "NN") for t in tokens]

print("CUDA available:", torch.cuda.is_available())

# -------------------------------
# Config
# -------------------------------
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

# -------------------------------------------------------------------------
# 2) Load model
# -------------------------------------------------------------------------
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!")

# -------------------------------------------------------------------------
# 3) Summarization
# -------------------------------------------------------------------------
prompt = (
    "Summarize the following scientific text in exactly 2 or 3 concise sentences, "
    "focusing only on the study’s purpose, methods, and main findings, "
    "keeping important medical/experimental details:\n\n"
    f"{original_text}\n\nSummary:"
)

messages = [{"role": "user", "content": prompt}]

# Build chat template text
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # harmless if model ignores it
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

# Slice off the prompt part
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# If Qwen thinking tokens are present, strip everything up to </think> (id 151668)
try:
    END_THINK = 151668
    index = len(output_ids) - output_ids[::-1].index(END_THINK)
except ValueError:
    index = 0

generated_summary = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n--- SUMMARY (MODEL) ---\n")
print(generated_summary)

# -------------------------------------------------------------------------
# 4) TF-IDF evaluation
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
# 5) ROUGE evaluation (model vs human)
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
# 6) Hallucination check (source-aware + human-aware) -- NLTK version
# -------------------------------------------------------------------------
print("\n--- HALLUCINATION (source + human) ---")

# Domain fillers you wanted to ignore
SCIENTIFIC_FILLERS = {
    "study","studies","analysis","analyses","model","compared","compare",
    "sheep","group","groups","reactive","nonreactive","control","pulmonary",
    "bal","bronchoalveolar","lavage","disease","airway","ascaris","antigen",
    "results","finding","findings","indicating","indicated","showed","observed",
    "parameters","tests","function","functions","measurement","measurements"
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
    return wordnet.NOUN  # fallback

def normalize_and_lemmatize(text: str):
    toks = word_tokenize(text)
    tagged = safe_pos_tag(toks)  # robust POS tagging
    lemmas = []
    for tok, tag in tagged:
        t = tok.lower()
        if t.isalpha() and t not in STOPWORDS:
            lemmas.append(wnl.lemmatize(t, pos=_to_wn_pos(tag)))
    return lemmas

# Build token lists/sets
src_tokens = normalize_and_lemmatize(original_text)
sum_tokens = normalize_and_lemmatize(generated_summary)
ref_tokens = normalize_and_lemmatize(human_summary)

src_set = set(src_tokens)
ref_set = set(ref_tokens)

# 1) Unsupported vs source (ignore scientific fillers)
unsupported_src = [w for w in sum_tokens if w not in src_set and w not in SCIENTIFIC_FILLERS]

# 2) Forgive words that appear in human reference
truly_unsupported = [w for w in unsupported_src if w not in ref_set]

# 3) Numeric hallucinations
src_numbers = set(re.findall(r"\d+(?:\.\d+)?", original_text.lower()))
sum_numbers = set(re.findall(r"\d+(?:\.\d+)?", generated_summary.lower()))
numeric_hallucinated = [n for n in sum_numbers if n not in src_numbers]

# 4) Score
unsupported_ratio = (len(truly_unsupported) / len(sum_tokens)) if len(sum_tokens) > 0 else 0.0
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
