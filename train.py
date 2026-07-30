from datasets import load_dataset
from colorama import Fore,init
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from multiprocessing import freeze_support
from trl import SFTTrainer,SFTConfig
from peft import LoraConfig,prepare_model_for_kbit_training,get_peft_model
import os
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# --- Kaggle token handling ---
# On Kaggle, prefer the Secrets Add-on instead of a .env file.
# Add "HUGGING_FACE_TOKEN" under Add-ons > Secrets in the notebook editor.
try:
    from kaggle_secrets import UserSecretsClient
    HUGGING_FACE_TOKEN = UserSecretsClient().get_secret("HUGGING_FACE_TOKEN")
except Exception:
    # fallback: environment variable (e.g. if set manually or via .env)
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

if HUGGING_FACE_TOKEN is None:
    raise ValueError(
        "HUGGING_FACE_TOKEN not found. Add it via Kaggle Secrets "
        "(Add-ons > Secrets) or set it as an environment variable."
    )

init(autoreset=True)

# --- Kaggle paths ---
DATA_PATH = "/kaggle/input/datasets/sahithiakulaa/instruction-jsonl/instruction.jsonl"
WORK_DIR = "/kaggle/working"
CHECKPOINT_DIR = os.path.join(WORK_DIR, "qlora", "checkpoints")
COMPLETE_CHECKPOINT_DIR = os.path.join(WORK_DIR, "qlora", "complete_checkpoint")
FINAL_MODEL_DIR = os.path.join(WORK_DIR, "qlora", "final_model")
CACHE_DIR = os.path.join(WORK_DIR, "hf_cache")

def main():
    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train"
    )
    print(Fore.YELLOW+str(dataset[2]))

    def format_chat_template(batch,tokenizer):
        system_prompt="""You are a helpful, honest and harmless assistant designed to help engineers.
        Think through each question logically and provide an answer. Don't make things up, if you're
        unable to answer a question, advise the  user that you're unable to asnwer as it is outside 
        of your scope."""
        
        # Apply chat template and append the result to the list
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}" \
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] " \
        "| trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}" \
        "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

        samples=[]
        questions=batch["question"]
        answers=batch["answer"]
        for i in range(len(questions)):
            row_json=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":questions[i]},
                {"role":"assistant","content":answers[i]}
            ]

            text=tokenizer.apply_chat_template(row_json,tokenize=False)
            samples.append(text)

        return {
                "instruction":questions,
                "response":answers,
                "text":samples
        }

    base_model="meta-llama/Llama-3.2-1B-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=HUGGING_FACE_TOKEN
    )
    def preprocess(batch):
        return format_chat_template(batch, tokenizer)

    full_dataset=dataset.map(
        preprocess,
        batched=True,
        batch_size=10
    )
    print(Fore.LIGHTMAGENTA_EX+str(full_dataset[0]))

    # Held-out split: used both for eval_loss during training AND as the
    # source of unseen benchmark questions for eval.py afterward.
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Save held-out questions so eval.py can compare baseline vs fine-tuned
    # on data the model genuinely never saw during training.
    held_out_questions = eval_dataset["instruction"][:20]  # cap for eval speed
    import json
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(os.path.join(WORK_DIR, "held_out_questions.json"), "w", encoding="utf-8") as f:
        json.dump(held_out_questions, f, ensure_ascii=False, indent=2)

    quant_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model=AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        token=HUGGING_FACE_TOKEN,
        cache_dir=CACHE_DIR,
    )

    print(Fore.CYAN+str(model))
    print(Fore.LIGHTYELLOW_EX+str(next(model.parameters())))

    

    peft_config=LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM"
    )
    model.gradient_checkpointing_enable()
    model=prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Force any stray bf16 params (e.g. lm_head/embed_tokens) to fp16
    # so they're compatible with fp16 GradScaler on T4/P100
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    trainer=SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=CHECKPOINT_DIR,
            dataset_text_field="text",
            max_seq_length=256,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,

            logging_steps=10,

            eval_strategy="steps",
            eval_steps=500,

            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,

            report_to="none",

            fp16=True,
            bf16=False,
        )
    )

    if os.path.isdir(CHECKPOINT_DIR) and any(
        d.startswith("checkpoint-") for d in os.listdir(CHECKPOINT_DIR)
    ):
        print("Resuming from latest checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting new training...")
        trainer.train()

    trainer.save_model(COMPLETE_CHECKPOINT_DIR)
    trainer.model.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    freeze_support()
    main()