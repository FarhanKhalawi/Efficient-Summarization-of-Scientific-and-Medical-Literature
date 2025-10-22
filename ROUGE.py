# summarize_and_eval.py
# -----------------------------------------------------------
# Generates summaries for all rows in a CSV and computes ROUGE.
# If CSV contains a 'REFERENCE' column (recommended), ROUGE is
# computed against it. Otherwise, a proxy ROUGE is computed
# against the full source TEXT (NOT a valid evaluation).
#-----------------------------------------------------------
#sudo apt update
#sudo apt install -y python3 python3-venv python3-pip
#python3 -V                    # sanity check
#python3 -m venv .venv
#source .venv/bin/activate
#python -m pip install --upgrade pip
#pip install torch transformers evaluate rouge-score pandas
#python -m pip install --upgrade pip setuptools wheel
#pip install "bitsandbytes>=0.43.3" accelerate
# -----------------------------------------------------------

import os
import re
import json
import argparse
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Optional but recommended:
# pip install evaluate rouge-score
try:
    import evaluate
    _HAS_EVALUATE = True
except Exception:
    _HAS_EVALUATE = False

# ----------------- Config -----------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
CSV_PATH   = "data/MeDAL/pretrain_subset/test.csv"  # expects TEXT column (+ REFERENCE if you have human gold)
OUT_DIR    = "outputs"
OUT_PRED_CSV = os.path.join(OUT_DIR, "pred_summaries.csv")
OUT_ROUGE_JSON = os.path.join(OUT_DIR, "rouge_results.json")

# Generation hyperparameters
GEN_TEMP = 0.4
GEN_TOPP = 0.9
MAX_NEW_TOKENS = 320
KEEP_TOKENS_FOR_ANSWER = 512

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

# 4-bit quantization for large model on a single GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ----------------- Helpers -----------------
def build_inputs(tok, system_msg, user_text):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text},
    ]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages, chat

def safe_trim_to_first_n_sentences(text, n=4, min_keep=3):
    text = text.replace("<think>", "").strip()
    text = re.sub(r"^\s*(?:Hmm[.,].*?\n+)+", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) > n:
        sents = sents[:n]
    if len(sents) < min_keep and len(text) > 0:
        return text
    return " ".join(sents).strip()

def postprocess(decoded: str) -> str:
    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1]
    m = re.search(r"<summary>(.*?)</summary>", decoded, flags=re.DOTALL | re.IGNORECASE)
    final = (m.group(1) if m else decoded).strip()
    if final.strip() in {"", "...", "…"}:
        return ""
    return safe_trim_to_first_n_sentences(final, n=4, min_keep=3)

@torch.inference_mode()
def generate_summary(model, tok, chat, keep_tokens_for_answer=512, source_text=None):
    # tokenize without truncation first to inspect length
    inputs = tok([chat], return_tensors="pt", truncation=False).to("cuda")
    max_ctx = getattr(model.config, "max_position_embeddings", 32768)

    # If too long, keep the HEAD of the abstract to leave room for generation
    if inputs.input_ids.shape[1] > (max_ctx - keep_tokens_for_answer) and source_text is not None:
        enc = tok(source_text, return_tensors="pt", truncation=False)
        keep = max_ctx - keep_tokens_for_answer - 128
        enc_ids = enc.input_ids[0, :keep].unsqueeze(0)
        trimmed = tok.decode(enc_ids[0], skip_special_tokens=True)
        _, chat = build_inputs(tok, SYSTEM_MSG_TAGGED, trimmed)
        inputs = tok([chat], return_tensors="pt").to("cuda")

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GEN_TEMP,
        top_p=GEN_TOPP,
    )
    new_tokens = out[0][len(inputs.input_ids[0]):]
    decoded = tok.decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return decoded

def compute_rouge(preds, refs, use_stemmer=True):
    """
    Computes ROUGE-1/2/L/Lsum (F1 by default) using evaluate. Returns dict.
    If evaluate isn't installed, returns {} and prints a message.
    """
    if not _HAS_EVALUATE:
        print("\n[Warning] The 'evaluate' package is not installed. "
              "Install with: pip install evaluate rouge-score")
        return {}
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=preds, references=refs, use_stemmer=use_stemmer)

# ----------------- Main -----------------
def main(limit_rows=None, skip_generation=False):
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. This script expects a CUDA GPU.")

    # Load model/tokenizer
    print("Loading tokenizer & model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.eval()

    # Read data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "TEXT" not in df.columns:
        raise ValueError("CSV must include a TEXT column containing the abstract.")

    if limit_rows is not None:
        df = df.head(limit_rows).copy()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Generate predictions unless asked to skip
    if not skip_generation:
        preds = []
        for i, row in df.iterrows():
            source_text = str(row["TEXT"]).strip()
            _, chat = build_inputs(tok, SYSTEM_MSG_TAGGED, source_text)
            decoded = generate_summary(model, tok, chat,
                                       keep_tokens_for_answer=KEEP_TOKENS_FOR_ANSWER,
                                       source_text=source_text)
            summary = postprocess(decoded)

            # fallback path if empty
            if not summary:
                _, chat_fb = build_inputs(tok, SYSTEM_MSG_FALLBACK, source_text)
                decoded_fb = generate_summary(model, tok, chat_fb,
                                              keep_tokens_for_answer=KEEP_TOKENS_FOR_ANSWER,
                                              source_text=source_text)
                summary = postprocess(decoded_fb) or ""

            preds.append(summary)

        df["PRED_SUMMARY"] = preds
        df.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8")
        print(f"\nSaved predictions to: {OUT_PRED_CSV}")
    else:
        if "PRED_SUMMARY" not in df.columns:
            raise ValueError("skip_generation=True but no PRED_SUMMARY column present in CSV.")
        print("\nSkipping generation step and using existing PRED_SUMMARY column.")

    # ROUGE evaluation
    use_proxy = False
    if "REFERENCE" in df.columns and df["REFERENCE"].notna().any():
        refs = df["REFERENCE"].fillna("").tolist()
    else:
        use_proxy = True
        # Proxy: compare to original TEXT (NOT a valid evaluation)
        refs = df["TEXT"].fillna("").tolist()

    preds = df["PRED_SUMMARY"].fillna("").tolist()

    scores = compute_rouge(preds, refs, use_stemmer=True)
    if scores:
        with open(OUT_ROUGE_JSON, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"\nROUGE results: {json.dumps(scores, indent=2)}")
        print(f"Saved ROUGE to: {OUT_ROUGE_JSON}")

    if use_proxy:
        print("\n[IMPORTANT] You did NOT provide a 'REFERENCE' column. "
              "The ROUGE above was computed against the full source TEXT as a proxy. "
              "This is NOT a valid evaluation. Add human reference summaries in a 'REFERENCE' column "
              "for meaningful ROUGE (recommended: report rouge1/rouge2/rougeL/rougeLsum F1).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_rows", type=int, default=None,
                        help="Optionally evaluate only the first N rows (for quick testing).")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation and only compute ROUGE using existing PRED_SUMMARY.")
    args = parser.parse_args()
    main(limit_rows=args.limit_rows, skip_generation=args.skip_generation)
