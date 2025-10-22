# summarize_one.py
import os, re, pandas as pd, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
CSV_PATH = "data/MeDAL/pretrain_subset/test.csv"  # adjust if needed

# ---- 4-bit quantized load ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def build_inputs(tok, system_msg, user_text):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text},
    ]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages, chat

def safe_trim_to_first_n_sentences(text, n=4, min_keep=3):
    text = text.replace("<think>", "").strip()
    # drop any leading "Hmm, ..." narration
    text = re.sub(r"^\s*(?:Hmm[.,].*?\n+)+", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) > n:
        sents = sents[:n]
    # ensure at least 3 sentences if possible
    if len(sents) < min_keep and len(text) > 0:
        return text
    return " ".join(sents).strip()

def postprocess(decoded: str) -> str:
    # 1) drop hidden reasoning: keep only after the LAST </think>
    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1]
    # 2) extract inside <summary>...</summary> if present
    m = re.search(r"<summary>(.*?)</summary>", decoded, flags=re.DOTALL | re.IGNORECASE)
    final = (m.group(1) if m else decoded).strip()
    # 3) normalize ellipses-only or empty
    if final.strip() in {"", "...", "…"}:
        return ""
    return safe_trim_to_first_n_sentences(final, n=4, min_keep=3)

def generate_summary(model, tok, chat, keep_tokens_for_answer=512, source_text=None):
    # ensure room in context (prefer keeping the HEAD of abstracts)
    inputs = tok([chat], return_tensors="pt", truncation=False).to("cuda")
    max_ctx = getattr(model.config, "max_position_embeddings", 32768)
    if inputs.input_ids.shape[1] > (max_ctx - keep_tokens_for_answer) and source_text is not None:
        # keep the HEAD (often abstracts put key context up front)
        enc = tok(source_text, return_tensors="pt", truncation=False)
        keep = max_ctx - keep_tokens_for_answer - 128
        enc_ids = enc.input_ids[0, :keep].unsqueeze(0)
        trimmed = tok.decode(enc_ids[0], skip_special_tokens=True)
        messages, chat = build_inputs(tok, SYSTEM_MSG_TAGGED, trimmed)
        inputs = tok([chat], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=320,       # a bit more room than before
            do_sample=True,
            temperature=GEN_TEMP,
            top_p=GEN_TOPP,
        )

    new_tokens = out[0][len(inputs.input_ids[0]):]
    decoded = tok.decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return decoded

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected.")

print("Loading tokenizer & model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16,
)
model.eval()

# ---- read first row ----
df = pd.read_csv(CSV_PATH)  # expects: ABSTRACT_ID, TEXT, LOCATION, LABEL
source_text = str(df.iloc[0]["TEXT"]).strip()

# ---- attempt 1: TAGGED prompt (preferred) ----
SYSTEM_MSG_TAGGED = (
    "Summarize the user's medical abstract in 3–4 sentences. "
    "Be clear and factual. Keep key clinical details (condition, intervention, measurements, outcomes). "
    "Return ONLY the summary wrapped exactly as:\n<summary>...</summary>\nNo preface, no analysis, no extra text."
)
GEN_TEMP, GEN_TOPP = 0.4, 0.9  # a little more deterministic

messages, chat = build_inputs(tok, SYSTEM_MSG_TAGGED, source_text)
print("Generating summary (attempt 1)...")
decoded = generate_summary(model, tok, chat, keep_tokens_for_answer=512, source_text=source_text)
summary = postprocess(decoded)

# ---- fallback: untagged prompt with slightly higher temperature ----
if not summary:
    SYSTEM_MSG_FALLBACK = (
        "Summarize the user's medical abstract in 3–4 sentences. "
        "Be clear and factual. Keep key clinical details (condition, intervention, measurements, outcomes). "
        "Output ONLY the 3–4 sentence summary—no preface, no analysis."
    )
    GEN_TEMP, GEN_TOPP = 0.7, 0.95
    messages, chat = build_inputs(tok, SYSTEM_MSG_FALLBACK, source_text)
    print("Generating summary (attempt 2, fallback)...")
    decoded_fb = generate_summary(model, tok, chat, keep_tokens_for_answer=512, source_text=source_text)
    summary = postprocess(decoded_fb)

# ---- last safety: if still empty, print a small debug preview ----
if not summary:
    preview = decoded[:400].replace("\n", " ")
    print("\n[Debug] Model raw (first 400 chars):", preview)
    summary = "Summary unavailable: model returned no usable text. Try re-running or adjust temperature/top_p."

print("\n--- SUMMARY ---\n")
print(summary)

os.makedirs("outputs", exist_ok=True)
with open("outputs/summary_first_test.txt", "w", encoding="utf-8") as f:
    f.write(summary + "\n")
print("\nSaved to outputs/summary_first_test.txt")

