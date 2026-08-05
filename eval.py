"""
eval.py — Base vs. Fine-Tuned (QLoRA) evaluation harness.

Reproduces the exact held-out split created in train.py (same seed=42
train_test_split) and evaluates the base Llama-3.2-1B-Instruct model
against the LoRA-adapted fine-tuned model on those unseen questions.

Both models are generated under a hard, identical token cap (default 200,
via --max-new-tokens). Generation is aborted the instant either model
crosses that many new tokens, enforced with an explicit StoppingCriteria
(not just the soft `max_new_tokens` kwarg), so neither model gets a longer
runway than the other and a single runaway/looping generation can't stall
the whole eval. Truncated generations aren't dropped — they're scored and
flagged (`truncated=True`, `tokens_generated=N`) so a cut-off answer is
never silently compared as if it were complete.

IMPORTANT — quality vs. behaviour are reported separately:
Because the training run didn't emit EOS tokens, every generation (base
*and* fine-tuned) runs until the hard 200-token cap rather than stopping
naturally. That's a generation-*behaviour* fact, not a content-quality
signal, so this harness deliberately keeps the two apart:

  - "Generation Quality" = accuracy, helpfulness, fluency, judged only on
    the content that was actually produced, via LLM-as-judge. The judge is
    explicitly instructed not to penalize a response for being cut off —
    it grades the substance of what's there, not whether it finished.
    `length_ratio` (generated length vs. gold length) has been removed
    entirely as a metric, since under a shared truncation cap it mostly
    measures the cap, not answer quality.
  - "Generation Behaviour" = truncation rate, average tokens generated,
    repetition rate. These describe *how* the model generates (a known,
    already-diagnosed limitation from missing EOS during training) and are
    reported side by side with quality, never folded into it.

What it measures:
  1. Generation quality (LLM-as-judge, content-only): accuracy,
     helpfulness, fluency — 1-5 scores per model, judged without penalty
     for token-cap truncation.
  2. Generation behaviour: truncation rate (fraction of generations that
     hit the 200-token cap before naturally emitting EOS), average tokens
     generated, and repetition rate (fraction of generations with a
     degenerate repeated n-gram).
  3. Auxiliary automatic text-overlap metrics vs. the gold answer
     (ROUGE-L, BLEU, language match) — kept in the CSV / a supplementary
     report section for reference, but not treated as the quality signal.

Outputs:
  - eval_results.csv        row per question, both model outputs + all metrics
  - eval_report.md          human-readable side-by-side report + summary table
  - eval_summary.json       aggregate numbers only (for CI / tracking over time)

Usage — Kaggle/cloud GPU, evaluating on the reproduced held-out split:

    python eval.py \
        --data-path /kaggle/input/datasets/sahithiakula/instruction-jsonl/instruction.jsonl \
        --adapter-path /kaggle/working/qlora/final_model \
        --num-samples 20

    # or evaluate the pushed HF hub adapter instead of a local checkpoint:
    python eval.py --adapter-path sahiithiii/Fine-Tuning-Llama-w-Multilingual-Dataset-using-qLoRA/final_model

Usage — local machine, quick smoke test on a handful of hand-picked examples,
no GPU required (see test_samples.json for the expected format):

    python eval.py \
        --test-samples test_samples.json \
        --adapter-path ./final_model \
        --no-4bit \
        --max-new-tokens 100

LLM-as-judge (accuracy / helpfulness / fluency) is ON by default, since it's
the metric that actually answers "is the content good" independent of the
token cap. Pass --no-judge to skip it (e.g. no judge model reachable) — the
report will still run, just without the Generation Quality numbers.

Note: meta-llama/Llama-3.2-1B-Instruct is a gated HF model, so
HUGGING_FACE_TOKEN must be set (as an env var locally, or via Kaggle
Secrets) either way — a local run doesn't get you around that.
"""

import argparse
import json
import os
import re
from collections import Counter

import torch
from colorama import Fore, init
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

init(autoreset=True)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = """You are a helpful, honest and harmless assistant designed to help engineers.
Think through each question logically and provide an answer. Don't make things up, if you're
unable to answer a question, advise the user that you're unable to asnwer as it is outside
of your scope."""

CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] "
    "| trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


class MaxNewTokensStoppingCriteria(StoppingCriteria):
    """
    Explicit, model-agnostic hard stop: once `max_new_tokens` tokens have been
    generated (counted from `start_length`, the length of the prompt), abort
    generation immediately — regardless of what either model is doing (still
    mid-sentence, ignoring EOS, looping, etc). This is applied identically to
    the base model and the fine-tuned model so the 200-token cap is a level
    playing field for comparison, not just a soft default on max_new_tokens.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return (input_ids.shape[-1] - self.start_length) >= self.max_new_tokens


import os
HUGGING_FACE_TOKEN=os.getenv("HUGGING_FACE_TOKEN")


# --------------------------------------------------------------------------
# Data: reproduce the exact held-out split from train.py
# --------------------------------------------------------------------------

def load_held_out(data_path: str, num_samples: int, held_out_questions_path: str | None):
    dataset = load_dataset("json", data_files=data_path, split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)  # same seed as train.py
    eval_dataset = split["test"]

    questions = eval_dataset["question"][:num_samples]
    answers = eval_dataset["answer"][:num_samples]

    # Sanity check against train.py's saved held_out_questions.json, if present,
    # to confirm this is really the same unseen split the model never trained on.
    if held_out_questions_path and os.path.exists(held_out_questions_path):
        with open(held_out_questions_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved[: len(questions)] != list(questions)[: len(saved)]:
            print(
                Fore.RED
                + "WARNING: held_out_questions.json does not match the split reproduced "
                "here (different dataset version or seed?). Proceeding anyway, but "
                "double check --data-path matches what train.py used."
            )
        else:
            print(Fore.GREEN + "Held-out split verified against held_out_questions.json.")

    return list(questions), list(answers)


def load_test_samples(path: str):
    """
    Local/offline alternative to reproducing the full dataset split — for
    quick smoke tests, or for a curated, category-balanced held-out set.

    Expects a JSON file containing a list of objects with "question" and
    "answer" keys, e.g.:

        [
          {"question": "How do you say 'good morning' in Hindi?", "answer": "सुप्रभात"},
          {"question": "What is a variable in programming?", "answer": "..."}
        ]

    ("instruction"/"response" keys are also accepted, to match the field
    names train.py renames the JSONL into internally.)

    Optional "category" and "language" keys (e.g. from test_samples_balanced.json)
    are picked up automatically if present and used to break results down by
    category/language in the report — that's the whole point of curating a
    stratified set instead of a random slice of the training distribution.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions, answers, categories, languages = [], [], [], []
    for item in data:
        q = item.get("question", item.get("instruction"))
        a = item.get("answer", item.get("response"))
        if q is None or a is None:
            raise ValueError(
                f"Each item in {path} needs a question/answer (or instruction/response) pair. Got: {item}"
            )
        questions.append(q)
        answers.append(a)
        categories.append(item.get("category"))
        languages.append(item.get("language"))
    return questions, answers, categories, languages


def load_tokenizer(token: str):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=token)
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(token: str, use_4bit: bool = True):
    """
    use_4bit=True requires a CUDA GPU + bitsandbytes (Kaggle/Colab/cloud GPU box).
    On a local machine with no GPU (or an Apple Silicon Mac), pass use_4bit=False
    to load in fp32/fp16 on CPU instead — slower, but it doesn't need bitsandbytes
    or CUDA at all. 1B params is small enough to be workable on CPU for a
    handful of --test-samples.
    """
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit quantized loading requires a CUDA GPU, but none was detected. "
                "Re-run with --no-4bit to load on CPU instead (slower, but works "
                "on a laptop for a small --test-samples run)."
            )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            token=token,
        )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, token=token)
    return model.to(device)


def load_finetuned_model(base_model, adapter_path: str):
    return PeftModel.from_pretrained(base_model, adapter_path)


@torch.no_grad()
def generate(model, tokenizer, question: str, max_new_tokens: int = 200) -> dict:
    """
    Generate a response, hard-capped at `max_new_tokens` for BOTH models so the
    comparison is apples-to-apples. Reports how many tokens were actually
    generated and whether the cap (rather than natural EOS) is what stopped it,
    so truncated generations can be flagged and evaluated accordingly rather
    than silently compared as if they were complete answers.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList(
        [MaxNewTokensStoppingCriteria(start_length=prompt_len, max_new_tokens=max_new_tokens)]
    )

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,      # transformers' own cap (belt...)
        stopping_criteria=stopping_criteria,  # ...and explicit hard stop (suspenders)
        do_sample=False,  # deterministic, for reproducible eval
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = output[0][prompt_len:]
    tokens_generated = new_tokens.shape[0]
    ended_with_eos = bool(tokens_generated > 0 and new_tokens[-1].item() == tokenizer.eos_token_id)
    truncated = (tokens_generated >= max_new_tokens) and not ended_with_eos

    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"text": text, "tokens_generated": tokens_generated, "truncated": truncated}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _tokenize(text: str):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 via LCS, dependency-free implementation."""
    ref, hyp = _tokenize(reference), _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec)


def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Simple sentence-level BLEU with uniform n-gram weights and brevity penalty."""
    ref, hyp = _tokenize(reference), _tokenize(hypothesis)
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
        precisions.append(overlap / sum(hyp_ngrams.values()))
    if min(precisions) == 0:
        return 0.0
    geo_mean = sum(p and __import__("math").log(p) for p in precisions) / max_n
    import math
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return bp * math.exp(geo_mean)


def detect_lang(text: str) -> str:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text) if text.strip() else "unknown"
    except Exception:
        return "unknown"


def repetition_flag(text: str, n: int = 4, min_repeats: int = 3) -> bool:
    """Flags degenerate output: same n-gram repeated min_repeats+ times."""
    toks = _tokenize(text)
    if len(toks) < n * min_repeats:
        return False
    ngrams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return any(c >= min_repeats for c in ngrams.values())


def score_pair(reference: str, hypothesis: str) -> dict:
    """
    Automatic, judge-free metrics. NOTE: `length_ratio` has been deliberately
    removed here. Under a shared hard truncation cap (both models stopped at
    the same token budget because EOS wasn't learned during training),
    generated-vs-gold length mostly reflects the cap, not answer quality —
    so it's not computed and not reported anywhere downstream.
    """
    ref_lang, hyp_lang = detect_lang(reference), detect_lang(hypothesis)
    return {
        "rouge_l": round(rouge_l(reference, hypothesis), 4),
        "bleu": round(bleu(reference, hypothesis), 4),
        "ref_lang": ref_lang,
        "hyp_lang": hyp_lang,
        "lang_match": ref_lang == hyp_lang,
        "repetitive": repetition_flag(hypothesis),
    }


# --------------------------------------------------------------------------
# LLM-as-judge (reuses litellm, already in requirements.txt)
#
# This is the primary "Generation Quality" signal (accuracy / helpfulness /
# fluency). It's explicitly instructed to grade only the content that's
# present and never penalize a response for being cut off by the token cap —
# generation behaviour (truncation, length, repetition) is tracked and
# reported separately, not folded into these scores.
# --------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading two AI responses (A and B) to the same question, against a reference answer.

Question: {question}
Reference answer: {reference}

Response A: {response_a}{a_trunc_note}

Response B: {response_b}{b_trunc_note}

IMPORTANT: Some responses were cut off mid-generation because they hit a hard
output-length limit (a known generation-length limitation, unrelated to the
model's knowledge or reasoning quality). Do NOT penalize a response for being
incomplete, cut off, or ending abruptly if it is marked "[TRUNCATED]" above.
Judge only the substance of the content that is actually present: is what
was said accurate, helpful, and fluent, as far as it goes? A truncated
response that is accurate and well-formed as far as it goes should score
the same as a complete response of equivalent quality — do not dock points
simply because it stops short.

Score each response from 1-5 on: accuracy (matches the facts/intent of the reference,
judged only on content present), helpfulness (judged only on content present), and
fluency/language-correctness (correct language, grammatical, judged only on content present).
Then say which response is better overall on that same content-only basis: A, B, or TIE.

Respond ONLY as compact JSON:
{{"a_accuracy":int,"a_helpfulness":int,"a_fluency":int,"b_accuracy":int,"b_helpfulness":int,"b_fluency":int,"winner":"A"|"B"|"TIE"}}"""


def llm_judge(
    question: str,
    reference: str,
    response_a: str,
    response_b: str,
    judge_model: str,
    truncated_a: bool = False,
    truncated_b: bool = False,
) -> dict:
    from litellm import completion
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        response_a=response_a,
        response_b=response_b,
        a_trunc_note=" [TRUNCATED — cut off by the output-length cap, not by the model choosing to stop]" if truncated_a else "",
        b_trunc_note=" [TRUNCATED — cut off by the output-length cap, not by the model choosing to stop]" if truncated_b else "",
    )
    try:
        resp = completion(model=judge_model, messages=[{"role": "user", "content": prompt}])
        content = resp["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        return json.loads(match.group(0)) if match else {}
    except Exception as e:
        print(Fore.RED + f"Judge call failed: {e}")
        return {}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs. QLoRA fine-tuned model.")
    parser.add_argument("--data-path", default="D:/qLoRA/data/instruction.jsonl",
                         help="Full instruction JSONL, used to reproduce train.py's held-out split. "
                              "Ignored if --test-samples is given.")
    parser.add_argument("--test-samples", default=None,
                         help="Path to a small local JSON file of [{question, answer}, ...] to eval on "
                              "instead of reproducing the full dataset split. Use this for a quick local "
                              "smoke test — see the docstring for the expected format.")
    parser.add_argument("--adapter-path", default="D:/qLoRA/final_model",
                         help="Local path or HF hub repo id of the fine-tuned LoRA adapter.")
    parser.add_argument("--held-out-questions", default="D:/qLoRA/data/held_out_questions.json")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=200,
                         help="Hard cap on generated tokens, enforced identically for both models. "
                              "Generation Quality scoring is content-only and does not penalize hitting "
                              "this cap; it's tracked separately under Generation Behaviour.")
    parser.add_argument("--output-dir", default="./eval_out")
    parser.add_argument("--no-4bit", action="store_true",
                         help="Load in fp16/fp32 instead of 4-bit bitsandbytes quantization. Use this on "
                              "machines without a CUDA GPU (laptops, CPU-only, Apple Silicon).")
    parser.add_argument("--no-judge", action="store_true",
                         help="Disable LLM-as-judge scoring. Judge is ON by default, since accuracy / "
                              "helpfulness / fluency (the Generation Quality metrics) come from it — "
                              "ROUGE-L/BLEU alone can't judge factual correctness. Use this flag only if "
                              "no judge model is reachable; the report will still run, minus those scores.")
    parser.add_argument("--judge-model", default="ollama_chat/llama3.1:latest",
                         help="litellm model string for the judge, e.g. 'gpt-4o-mini' or an Ollama model.")
    args = parser.parse_args()
    args.use_judge = not args.no_judge

    os.makedirs(args.output_dir, exist_ok=True)
    token = HUGGING_FACE_TOKEN

    if args.test_samples:
        print(Fore.CYAN + f"Loading local test samples from {args.test_samples}...")
        questions, answers, categories, languages = load_test_samples(args.test_samples)
    else:
        print(Fore.CYAN + "Loading held-out set (reproducing train.py's seed=42 split)...")
        questions, answers = load_held_out(args.data_path, args.num_samples, args.held_out_questions)
        categories, languages = [None] * len(questions), [None] * len(questions)
    print(Fore.CYAN + f"Evaluating on {len(questions)} questions.")

    if not args.use_judge:
        print(Fore.RED + "WARNING: --no-judge set. Generation Quality (accuracy/helpfulness/fluency) "
              "will NOT be available — only the auxiliary ROUGE-L/BLEU/repetition/truncation numbers.")

    print(Fore.CYAN + "Loading tokenizer + base model...")
    tokenizer = load_tokenizer(token)
    base_model = load_base_model(token, use_4bit=not args.no_4bit)

    print(Fore.CYAN + f"Generating BASE model responses (hard cap: {args.max_new_tokens} tokens)...")
    base_gens = [generate(base_model, tokenizer, q, args.max_new_tokens) for q in questions]
    for i, g in enumerate(base_gens):
        if g["truncated"]:
            print(Fore.RED + f"  [base] Q{i} hit the {args.max_new_tokens}-token cap before finishing "
                  "(generation-behaviour note only — content is still judged on its own merits).")

    print(Fore.CYAN + f"Attaching LoRA adapter from {args.adapter_path}...")
    ft_model = load_finetuned_model(base_model, args.adapter_path)

    print(Fore.CYAN + f"Generating FINE-TUNED model responses (hard cap: {args.max_new_tokens} tokens)...")
    ft_gens = [generate(ft_model, tokenizer, q, args.max_new_tokens) for q in questions]
    for i, g in enumerate(ft_gens):
        if g["truncated"]:
            print(Fore.RED + f"  [finetuned] Q{i} hit the {args.max_new_tokens}-token cap before finishing "
                  "(generation-behaviour note only — content is still judged on its own merits).")

    rows = []
    for i, (q, ref, base_gen, ft_gen, cat, lang) in enumerate(
        zip(questions, answers, base_gens, ft_gens, categories, languages)
    ):
        base_out, ft_out = base_gen["text"], ft_gen["text"]

        # Truncated generations are still scored (not dropped), but truncation
        # itself is a behaviour flag, kept out of any quality metric. The judge
        # (below) is told which side was cut off so it grades content only.
        base_metrics = score_pair(ref, base_out)
        ft_metrics = score_pair(ref, ft_out)
        base_metrics["truncated"] = base_gen["truncated"]
        base_metrics["tokens_generated"] = base_gen["tokens_generated"]
        ft_metrics["truncated"] = ft_gen["truncated"]
        ft_metrics["tokens_generated"] = ft_gen["tokens_generated"]

        row = {
            "id": i,
            "category": cat,
            "language": lang,
            "question": q,
            "reference": ref,
            "base_response": base_out,
            "finetuned_response": ft_out,
            **{f"base_{k}": v for k, v in base_metrics.items()},
            **{f"ft_{k}": v for k, v in ft_metrics.items()},
        }

        if args.use_judge:
            verdict = llm_judge(
                q, ref, base_out, ft_out, args.judge_model,
                truncated_a=base_gen["truncated"], truncated_b=ft_gen["truncated"],
            )
            row["judge_verdict"] = verdict

        rows.append(row)
        log_line = (
            f"[{i+1}/{len(questions)}] done "
            f"(ROUGE-L base={base_metrics['rouge_l']:.2f} ft={ft_metrics['rouge_l']:.2f}, "
            f"lang_match base={base_metrics['lang_match']} ft={ft_metrics['lang_match']}, "
            f"truncated base={base_metrics['truncated']} ft={ft_metrics['truncated']}"
        )
        if args.use_judge and row.get("judge_verdict"):
            v = row["judge_verdict"]
            log_line += (
                f", judge acc/help/flu base={v.get('a_accuracy')}/{v.get('a_helpfulness')}/{v.get('a_fluency')} "
                f"ft={v.get('b_accuracy')}/{v.get('b_helpfulness')}/{v.get('b_fluency')}"
            )
        log_line += ")"
        print(Fore.YELLOW + log_line)

    # -------------------- aggregate + write outputs --------------------
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    df.to_csv(csv_path, index=False)

    def agg_behaviour(prefix):
        return {
            "truncation_rate": round(df[f"{prefix}_truncated"].mean(), 3),
            "avg_tokens_generated": round(df[f"{prefix}_tokens_generated"].mean(), 1),
            "repetition_rate": round(df[f"{prefix}_repetitive"].mean(), 3),
        }

    def agg_auxiliary(prefix):
        return {
            "avg_rouge_l": round(df[f"{prefix}_rouge_l"].mean(), 4),
            "avg_bleu": round(df[f"{prefix}_bleu"].mean(), 4),
            "lang_match_rate": round(df[f"{prefix}_lang_match"].mean(), 3),
        }

    def agg_quality():
        """
        Accuracy / helpfulness / fluency, averaged from judge verdicts.
        A = base, B = finetuned. Skips rows where the judge call failed.
        """
        if not args.use_judge or "judge_verdict" not in df.columns:
            return None
        verdicts = [v for v in df["judge_verdict"] if isinstance(v, dict) and v]
        if not verdicts:
            return None

        def avg(key):
            vals = [v[key] for v in verdicts if key in v]
            return round(sum(vals) / len(vals), 2) if vals else None

        return {
            "base": {
                "accuracy": avg("a_accuracy"),
                "helpfulness": avg("a_helpfulness"),
                "fluency": avg("a_fluency"),
            },
            "finetuned": {
                "accuracy": avg("b_accuracy"),
                "helpfulness": avg("b_helpfulness"),
                "fluency": avg("b_fluency"),
            },
            "n_judged": len(verdicts),
        }

    summary = {
        "n": len(df),
        "max_new_tokens_cap": args.max_new_tokens,
        "quality": agg_quality(),  # accuracy / helpfulness / fluency — content-only, truncation-blind
        "behaviour": {
            "base": agg_behaviour("base"),
            "finetuned": agg_behaviour("ft"),
        },
        "auxiliary_text_overlap": {
            "base": agg_auxiliary("base"),
            "finetuned": agg_auxiliary("ft"),
        },
    }

    # Per-category / per-language breakdown — only meaningful when the input
    # file actually carries that metadata (e.g. test_samples_balanced.json).
    # This is the point of curating a stratified set instead of a random
    # slice: an aggregate ROUGE-L can hide e.g. safety refusals collapsing
    # or Kannada lagging behind Hindi even while the overall average looks fine.
    breakdown = {}
    for dim in ("category", "language"):
        if df[dim].notna().any():
            dim_summary = {}
            for value, group in df.groupby(dim):
                dim_summary[str(value)] = {
                    "n": len(group),
                    "base_rouge_l": round(group["base_rouge_l"].mean(), 3),
                    "ft_rouge_l": round(group["ft_rouge_l"].mean(), 3),
                    "base_lang_match": round(group["base_lang_match"].mean(), 3),
                    "ft_lang_match": round(group["ft_lang_match"].mean(), 3),
                }
            breakdown[dim] = dim_summary
    if breakdown:
        summary["breakdown"] = breakdown

    if args.use_judge and "judge_verdict" in df.columns:
        winners = [v.get("winner") for v in df["judge_verdict"] if isinstance(v, dict) and v.get("winner")]
        summary["judge_winner_counts"] = dict(Counter(winners))

    with open(os.path.join(args.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # -------------------- markdown report --------------------
    md = ["# Base vs. Fine-Tuned Evaluation Report\n",
          f"Evaluated on **{summary['n']}** held-out (unseen) questions. "
          f"Generation hard-capped at **{summary['max_new_tokens_cap']} tokens** for both models "
          "(EOS wasn't learned during training, so every generation runs to this cap). "
          "Quality and generation behaviour are reported separately below — behaviour is a known "
          "limitation, not evidence the content itself is poor.\n"]

    md.append("## Generation Quality\n")
    md.append("_LLM-judge scores (1-5), content-only — the judge is instructed not to penalize "
               "responses for being cut off by the token cap._\n")
    if summary["quality"]:
        q = summary["quality"]
        md += [
            "| Metric | Base | Fine-tuned |",
            "|---|---|---|",
            f"| Accuracy | {q['base']['accuracy']} | {q['finetuned']['accuracy']} |",
            f"| Helpfulness | {q['base']['helpfulness']} | {q['finetuned']['helpfulness']} |",
            f"| Fluency | {q['base']['fluency']} | {q['finetuned']['fluency']} |",
            f"\n_(n={q['n_judged']} judged responses per model)_\n",
        ]
        if "judge_winner_counts" in summary:
            md.append(f"**Judge overall-preference counts (A=base, B=finetuned):** {summary['judge_winner_counts']}\n")
    else:
        md.append("_Run without `--no-judge` to get Accuracy / Helpfulness / Fluency scores here._\n")

    md.append("## Generation Behaviour\n")
    md.append("_Known limitation (missing EOS during training → every generation hits the token cap). "
               "Tracked here for visibility, not used to judge content quality above._\n")
    b_base, b_ft = summary["behaviour"]["base"], summary["behaviour"]["finetuned"]
    md += [
        "| Metric | Base | Fine-tuned |",
        "|---|---|---|",
        f"| Truncation rate (hit {summary['max_new_tokens_cap']}-tok cap) | {b_base['truncation_rate']} | {b_ft['truncation_rate']} |",
        f"| Average tokens generated | {b_base['avg_tokens_generated']} | {b_ft['avg_tokens_generated']} |",
        f"| Repetition rate (lower=better) | {b_base['repetition_rate']} | {b_ft['repetition_rate']} |",
        "",
    ]

    md.append("## Auxiliary text-overlap metrics\n")
    md.append("_ROUGE-L / BLEU / language match vs. the gold answer. Supplementary reference numbers "
               "only — not the quality signal (see Generation Quality above) and somewhat depressed "
               "by shared truncation, since they compare against a complete gold answer._\n")
    a_base, a_ft = summary["auxiliary_text_overlap"]["base"], summary["auxiliary_text_overlap"]["finetuned"]
    md += [
        "| Metric | Base | Fine-tuned |",
        "|---|---|---|",
        f"| ROUGE-L | {a_base['avg_rouge_l']} | {a_ft['avg_rouge_l']} |",
        f"| BLEU | {a_base['avg_bleu']} | {a_ft['avg_bleu']} |",
        f"| Language match rate | {a_base['lang_match_rate']} | {a_ft['lang_match_rate']} |",
        "",
    ]

    if "breakdown" in summary:
        for dim, dim_summary in summary["breakdown"].items():
            md.append(f"## By {dim}\n")
            md.append("| " + dim.capitalize() + " | n | ROUGE-L (base) | ROUGE-L (ft) | Lang match (base) | Lang match (ft) |")
            md.append("|---|---|---|---|---|---|")
            for value, stats in sorted(dim_summary.items()):
                md.append(
                    f"| {value} | {stats['n']} | {stats['base_rouge_l']} | {stats['ft_rouge_l']} | "
                    f"{stats['base_lang_match']} | {stats['ft_lang_match']} |"
                )
            md.append("")

    md.append("## Sample-by-sample comparison\n")
    for row in rows:
        base_trunc = " **[TRUNCATED at cap — not penalized]**" if row["base_truncated"] else ""
        ft_trunc = " **[TRUNCATED at cap — not penalized]**" if row["ft_truncated"] else ""
        tag = " / ".join(str(v) for v in (row.get("category"), row.get("language")) if v)
        header = f"### Q{row['id']}: {row['question']}"
        if tag:
            header += f"  `{tag}`"
        md.append(header + "\n")
        md.append(f"**Reference:** {row['reference']}\n")

        base_judge = ""
        ft_judge = ""
        if "judge_verdict" in row and isinstance(row["judge_verdict"], dict) and row["judge_verdict"]:
            v = row["judge_verdict"]
            base_judge = f", accuracy={v.get('a_accuracy')}, helpfulness={v.get('a_helpfulness')}, fluency={v.get('a_fluency')}"
            ft_judge = f", accuracy={v.get('b_accuracy')}, helpfulness={v.get('b_helpfulness')}, fluency={v.get('b_fluency')}"

        md.append(f"**Base:**{base_trunc} {row['base_response']}  \n"
                   f"_(tokens={row['base_tokens_generated']}{base_judge})_\n")
        md.append(f"**Fine-tuned:**{ft_trunc} {row['finetuned_response']}  \n"
                   f"_(tokens={row['ft_tokens_generated']}{ft_judge})_\n")
        if "judge_verdict" in row:
            md.append(f"**Judge winner:** {row['judge_verdict'].get('winner') if isinstance(row['judge_verdict'], dict) else row['judge_verdict']}\n")
        md.append("---\n")

    md_path = os.path.join(args.output_dir, "eval_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(Fore.GREEN + f"\nDone. Wrote:\n  {csv_path}\n  {md_path}\n  {os.path.join(args.output_dir, 'eval_summary.json')}")
    print(Fore.GREEN + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()