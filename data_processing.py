import json
import random
from collections import defaultdict

from datasets import load_dataset
from colorama import Fore, init

init(autoreset=True)

random.seed(42)

# -------------------------
# Configuration
# -------------------------

ANUDESH_LIMIT = 15000

LANGUAGE_TARGETS = {
    "hin_Deva": 10000,
    "tel_Telu": 9000,
    "tam_Taml": 8000,
    "kan_Knda": 6000,
    "eng_Latn": 2000,
}

LANGS = list(LANGUAGE_TARGETS.keys())

# -------------------------
# Storage
# -------------------------

anudesh_pairs = []
language_pairs = defaultdict(list)

# ============================================================
# 1. Load Anudesh
# ============================================================

print(Fore.GREEN + "\nLoading Anudesh")

ds = load_dataset(
    "ai4bharat/indic-align",
    "Anudesh",
    split="train"
)

for row in ds:

    interactions = row["interactions"]

    for pair in interactions:

        if len(pair) != 2:
            continue

        question, answer = pair

        if not question or not answer:
            continue

        question = question.strip()
        answer = answer.strip()

        if question == "" or answer == "":
            continue

        anudesh_pairs.append({
            "question": question,
            "answer": answer,
            "source": "Anudesh"
        })

print(Fore.CYAN + f"Collected {len(anudesh_pairs)} Anudesh pairs")

# ============================================================
# 2. Load Dolly + HHRLHF
# ============================================================

for config in ["Dolly_T", "HHRLHF_T"]:

    print(Fore.GREEN + f"\nLoading {config}")

    ds = load_dataset(
        "ai4bharat/indic-align",
        config,
        split="train"
    )

    available_languages = [
        lang
        for lang in LANGS
        if lang in ds.column_names
    ]

    print(Fore.YELLOW + str(available_languages))

    for row in ds:

        for lang in available_languages:

            conversations = row.get(lang)

            if not conversations:
                continue

            pair = conversations[0]

            if len(pair) != 2:
                continue

            question, answer = pair

            if not question or not answer:
                continue

            question = question.strip()
            answer = answer.strip()

            if question == "" or answer == "":
                continue

            language_pairs[lang].append({
                "question": question,
                "answer": answer,
                "language": lang,
                "source": config
            })

# ============================================================
# 3. Shuffle everything
# ============================================================

random.shuffle(anudesh_pairs)

for lang in language_pairs:
    random.shuffle(language_pairs[lang])

# ============================================================
# 4. Sample balanced dataset
# ============================================================

final_dataset = []

# Unknown-language instruction tuning
final_dataset.extend(anudesh_pairs[:ANUDESH_LIMIT])

# Explicit language balancing
for lang, target in LANGUAGE_TARGETS.items():

    available = len(language_pairs[lang])

    take = min(target, available)

    print(
        Fore.CYAN +
        f"{lang}: taking {take}/{available}"
    )

    final_dataset.extend(
        language_pairs[lang][:take]
    )

# ============================================================
# 5. Final shuffle
# ============================================================

random.shuffle(final_dataset)

# ============================================================
# 6. Save JSONL
# ============================================================

with open(
    "data/instruction.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for sample in final_dataset:

        f.write(
            json.dumps(
                {
                    "question": sample["question"],
                    "answer": sample["answer"]
                },
                ensure_ascii=False
            ) + "\n"
        )

# ============================================================
# 7. Statistics
# ============================================================

print(Fore.GREEN)
print("=" * 60)
print("Dataset Statistics")
print("=" * 60)

print(f"Total examples: {len(final_dataset)}")

source_count = defaultdict(int)

language_count = defaultdict(int)

for sample in final_dataset:

    source_count[sample["source"]] += 1

    if "language" in sample:
        language_count[sample["language"]] += 1
    else:
        language_count["Unknown (Anudesh)"] += 1

print("\nSource Distribution")

for k, v in source_count.items():
    print(f"{k:15} {v}")

print("\nLanguage Distribution")

for k, v in language_count.items():
    print(f"{k:20} {v}")

print(Fore.GREEN)
print("\nSaved to data/instruction.jsonl")