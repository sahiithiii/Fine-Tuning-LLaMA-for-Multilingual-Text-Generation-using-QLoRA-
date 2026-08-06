# qLoRA Multilingual Fine-Tuning Project

## Title

**Multilingual QLoRA Fine-Tuning of `meta-llama/Llama-3.2-1B-Instruct`**

---

# Overview

This repository contains a multilingual fine-tuning pipeline built on top of **`meta-llama/Llama-3.2-1B-Instruct`** using **QLoRA** and **PEFT**. It includes the complete training pipeline, evaluation framework, dataset preparation scripts, and LoRA adapter artifacts for multilingual instruction tuning.

The primary objective is to improve multilingual instruction-following capabilities while maintaining computational efficiency through parameter-efficient fine-tuning.

---

# Motivation

Large Language Models exhibit strong performance in English but often struggle with multilingual instruction following, particularly in low-resource Indian languages.

This project explores **QLoRA**, a parameter-efficient fine-tuning technique, to adapt Llama 3.2 for multilingual conversational tasks while significantly reducing GPU memory requirements through 4-bit quantization.

---

# Features

- QLoRA-based fine-tuning with PEFT
- Multilingual instruction tuning
- 4-bit NF4 quantization
- LoRA adapters
- Automatic held-out evaluation
- LLM-as-a-Judge evaluation
- Behavioural metrics analysis
- Reproducible train/test split
- Local and Kaggle-compatible training/evaluation pipeline

---

# Dataset

The project uses multilingual instruction-response pairs stored in JSONL format.

## Dataset Files

| File | Description |
|------|-------------|
| `instruction.jsonl` | Primary multilingual instruction dataset |
| `held_out_questions.json` | Saved evaluation split used during training |
| `test_samples_balanced.json` | Curated evaluation dataset |

## Languages

- English
- Hindi
- Kannada
- Tamil
- Telugu

## Evaluation Categories

- Coding
- General Knowledge
- Multilingual Factual QA
- Reasoning
- Safety
- Summarization
- Translation

---

# Model Architecture

| Component | Value |
|-----------|-------|
| Base Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Architecture | Decoder-only Transformer |
| Fine-tuning | LoRA (PEFT) |
| Quantization | 4-bit NF4 |
| Task | Causal Language Modeling |

### LoRA Configuration

| Parameter | Value |
|-----------|------:|
| Rank (r) | 32 |
| Alpha | 64 |
| Dropout | 0.05 |
| Task Type | CAUSAL_LM |

### Target Modules

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `up_proj`
- `down_proj`
- `gate_proj`

---

# Fine-Tuning Methodology

The model is trained using the **QLoRA** framework with PEFT.

Training pipeline:

1. Load the base model in **4-bit quantized** mode using `bitsandbytes`.
2. Prepare the model for k-bit training using `prepare_model_for_kbit_training`.
3. Attach LoRA adapters using `get_peft_model`.
4. Format training samples using the tokenizer's chat template.
5. Fine-tune using `trl.SFTTrainer`.

---

# Training Configuration

| Parameter | Value |
|-----------|------:|
| Base Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Quantization | 4-bit NF4 |
| Compute Type | FP16 |
| Epochs | 2 |
| Max Sequence Length | 256 |
| Learning Rate | 2e-5 |
| Train Batch Size | 2 |
| Eval Batch Size | 2 |
| Gradient Accumulation | 4 |
| Evaluation Strategy | Steps |
| Evaluation Interval | 500 |
| Save Interval | 500 |
| Saved Checkpoints | 3 |

---

# Hyperparameters

| Hyperparameter | Value |
|---------------|------:|
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.05 |
| Learning Rate | 2e-5 |
| Epochs | 2 |
| Max Sequence Length | 256 |
| Batch Size | 2 |
| Gradient Accumulation | 4 |
| Save Steps | 500 |
| Evaluation Steps | 500 |

---

# Evaluation

The evaluation framework compares the **base model** and the **fine-tuned model** on identical unseen multilingual questions.

## Evaluation Procedure

- Recreates the held-out split from training.
- Generates deterministic outputs (`do_sample=False`).
- Uses a hard **200-token generation cap**.
- Computes behavioural metrics independently from quality metrics.
- Uses an **LLM-as-a-Judge** to evaluate:
  - Accuracy
  - Helpfulness
  - Fluency

Additional evaluation options include:

- CPU fallback (`--no-4bit`)
- Custom evaluation dataset (`--test-samples`)

---

# Evaluation Results

## Evaluation Configuration

| Parameter | Value |
|-----------|------:|
| Evaluation Samples | 15 |
| Judged Samples | 11 |
| Maximum Generated Tokens | 200 |

---

## Overall Quality

| Metric | Base | Fine-Tuned |
|--------|------:|-----------:|
| Accuracy | 2.91 | **2.92** |
| Helpfulness | 3.18 | **3.58** |
| Fluency | 2.45 | **2.86** |

---

## Behaviour Metrics

| Metric | Base | Fine-Tuned |
|--------|------:|-----------:|
| Truncation Rate | 0.533 | 0.541 |
| Average Tokens Generated | 131.5 | **157.7** |
| Repetition Rate | 0.20 | 0.21 |

---

## Text Similarity Metrics

| Metric | Base | Fine-Tuned |
|--------|------:|-----------:|
| ROUGE-L | 0.3298 | **0.4218** |
| BLEU | 0.1419 | **0.2639** |
| Language Match Rate | 0.733 | **0.871** |

---

## Category-wise Performance

| Category | Samples | Base ROUGE-L | Fine-Tuned ROUGE-L |
|----------|--------:|-------------:|-------------------:|
| Coding | 2 | 0.073 | **0.087** |
| General Knowledge | 3 | 0.545 | **0.896** |
| Multilingual Factual QA | 2 | 0.514 | **0.622** |
| Reasoning | 2 | **0.327** | 0.218 |
| Safety | 2 | 0.233 | **0.276** |
| Summarization | 2 | 0.509 | **0.527** |
| Translation | 2 | 0.000 | 0.000 |

---

## Language-wise Performance

| Language | Samples | Base ROUGE-L | Fine-Tuned ROUGE-L |
|----------|--------:|-------------:|-------------------:|
| English | 2 | 0.426 | **0.499** |
| Hindi | 4 | **0.511** | 0.200 |
| Kannada | 3 | 0.275 | **0.283** |
| Tamil | 3 | **0.212** | 0.194 |
| Telugu | 3 | 0.196 | **0.226** |

---

## Judge Preference

The fine-tuned model was preferred in **8 out of 11** judged responses.

---

# Inference

Load the tokenizer and base model using the Hugging Face Transformers API, attach the LoRA adapter using PEFT, and generate responses using the tokenizer's chat template.

The repository supports both:

- CUDA-based inference with 4-bit quantization
- CPU inference using `--no-4bit`

---

# Repository Structure

```text
.
├── data/
│   ├── instruction.jsonl
│   ├── held_out_questions.json
│   └── test_samples_balanced.json
│
├── final_model/
├── complete_checkpoint/
├── eval_out/
│
├── train.py
├── eval.py
├── preprocessing.py
├── data_processing.py
├── synthetic_data_generation.py
│
├── requirements.txt
│
├── llama-fine-tuning-on-multilingual-dataset-w-qlora.ipynb
├── trained_model_op.ipynb
├── test.ipynb
│
└── Evaluation Report.pdf
```

---

# Reproducibility

The repository supports reproducible training and evaluation through:

- Fixed random seed
- Saved held-out evaluation split
- Deterministic evaluation pipeline
- Exact dependency versions

Key packages:

| Library | Version |
|----------|---------|
| Transformers | 4.46.3 |
| TRL | 0.11.4 |
| PEFT | 0.13.2 |
| Accelerate | 0.34.2 |
| BitsAndBytes | 0.43.3 |

Authentication for gated model access is handled using the `HUGGING_FACE_TOKEN` environment variable.

---

# Limitations

- Evaluation is performed on a relatively small held-out dataset.
- Hard 200-token cap can truncate longer responses.
- Translation performance remains limited.
- Some reasoning tasks did not improve consistently.
- Requires gated access to the Meta Llama model.

---

# Future Work

- Improve EOS handling to reduce truncation.
- Expand multilingual training data.
- Increase evaluation dataset size.
- Explore additional LoRA configurations.
- Evaluate on larger multilingual benchmarks.
- Extend support to larger Llama variants.

