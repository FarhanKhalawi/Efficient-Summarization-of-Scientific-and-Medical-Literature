# Efficient Summarization of Scientific and Medical Literature

This project investigates automatic summarization of scientific and biomedical literature using advanced natural language processing (NLP) techniques. The system generates short, coherent summaries of biomedical abstracts while explicitly focusing on factual consistency and the reduction of hallucinated content.


##  Project Overview

The rapid growth of scientific and medical literature makes it increasingly challenging for researchers and clinicians to remain up to date. While large language models can generate fluent summaries, they often produce hallucinations, statements that are not supported by the source text. This project explores the use of transformer-based models from the Qwen3 family to summarise biomedical abstracts and systematically analyse their summarisation quality and factual faithfulness.

Two models are evaluated:

- Qwen3-0.6B 

- Qwen3-Reranker-8B 

Both models are tested on biomedical abstracts from the MeDAL dataset.

### Key Features
- Generates concise summaries of biomedical abstracts (typically 2–3 sentences)
- Evaluates summary quality using ROUGE and TF-IDF cosine similarity
- Analyses hallucinations using NLI-based, heuristic, and LLM-based evaluation methods
- Provides a multi-metric framework for assessing factual consistency


##  Technologies Used
- **Python 3.10+**
- **Transformer-based models (Qwen3)**
- **PyTorch**
- **Hugging Face Transformers**
- **Jupyter Notebook**
- **ROUGE** for evaluation metrics
- **NLTK / Scikit-learn**