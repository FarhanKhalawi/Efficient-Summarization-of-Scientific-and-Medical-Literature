
import re
import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# -------------------------------
# NLTK (tokenization + lemmatization)
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

# Load data
test_df = pd.read_csv(TEST_CSV)
human_df = pd.read_csv(HUMAN_CSV)

# Load model once
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!")

# === Key fix: set GenerationConfig explicitly (no auto "modified" message) ===
base = model.generation_config
gen_cfg = GenerationConfig(
    # decoding
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=20,
    # lengths
    max_new_tokens=256,
    min_new_tokens=40,
    # tokens (explicit to avoid default-modified notices)
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=base.bos_token_id,
    eos_token_id=base.eos_token_id
)

# -------------------------------
# Hallucination helpers
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
# Run for first ten rows
# -------------------------------
for i in range(10):
    print(f"\n====================== SAMPLE {i+1} ======================\n")
    row = test_df.iloc[i]
    original_text = str(row["TEXT"])
    abstract_id = int(row["ABSTRACT_ID"])

    # Match human summary by ABSTRACT_ID
    ref_row = human_df[human_df["ABSTRACT_ID"] == abstract_id]
    if ref_row.empty:
        print(f"No human summary for ABSTRACT_ID={abstract_id}")
        continue
    human_summary = ref_row["HUMAN_SUMMARY"].iloc[0]

    # ---- Summarization ----
    prompt = (
        "Summarize the following scientific text in EXACTLY 2 or 3 sentences. "
        "ONLY use facts explicitly present in the text. "
        "If a number or unit is not present in the text, do NOT invent it. "
        "Focus strictly on purpose, methods, and main findings.\n\n"
        f"{original_text}\n\nSummary:"
    )

    messages = [{"role": "user", "content": prompt}]
    # Disable Qwen thinking so no <think> appears
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Truncate very long inputs instead of failing silently
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

    # Remove the prompt portion
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    generated_summary = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # Safety: strip any <think>...</think> if present
    generated_summary = re.sub(r"(?is)<think>.*?</think>\s*", "", generated_summary)
    generated_summary = re.sub(r"(?is)^.*?</think>\s*", "", generated_summary)

    # Enforce 2–3 sentences if the model drifts
    sents = sent_tokenize(generated_summary)
    if len(sents) > 3:
        generated_summary = " ".join(sents[:3])

    print("\n--- SUMMARY (MODEL) ---\n")
    print(generated_summary)

    # ---- TF-IDF ----
    corpus = [original_text, generated_summary]
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    print("\n--- TF-IDF SIMILARITY ---")
    print(f"Cosine similarity (original vs summary): {similarity:.4f}")

    # ---- ROUGE ----
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(human_summary, generated_summary)
    print("\n--- ROUGE (model vs human) ---")
    print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2 F1: {scores['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")

    # ---- Hallucination check ----
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
    rouge1_f1 = scores["rouge1"].fmeasure
    soften = 1.0 - min(rouge1_f1, 0.6) * 0.3
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
