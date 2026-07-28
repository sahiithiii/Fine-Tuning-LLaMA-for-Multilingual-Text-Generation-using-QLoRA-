from datasets import load_dataset
from colorama import Fore,init
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from multiprocessing import freeze_support
from trl import SFTTrainer,SFTConfig
from peft import LoraConfig,prepare_model_for_kbit_training,get_peft_model
import os
from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

if HUGGING_FACE_TOKEN is None:
    raise ValueError("HF_TOKEN not found in .env file")

init(autoreset=True)
def main():
    dataset = load_dataset(
        "json",
        data_files="data/instruction.jsonl",
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

    base_model="meta-llama/Llama-3.2-3B-Instruct"  #Llama-3.2-1B
    tokenizer=AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=HUGGING_FACE_TOKEN
    )
    def preprocess(batch):
        return format_chat_template(batch, tokenizer)

    train_dataset=dataset.map(
        preprocess,
        batched=True,
        batch_size=10
    )
    print(Fore.LIGHTMAGENTA_EX+str(train_dataset[0]))

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
        cache_dir="./workspace",
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

    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    trainer=SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir="/content/drive/MyDrive/qlora/checkpoints",
            max_length=512,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,

            logging_steps=10,

            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,

            fp16=True,
            bf16=False,
        )
    )
    dtypes = set(p.dtype for p in model.parameters() if p.requires_grad)
    print(dtypes)  # should show only torch.float32 and/or torch.float16

    checkpoint_dir = "/content/drive/MyDrive/qlora/checkpoints"

    if os.path.isdir(checkpoint_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(checkpoint_dir)
    ):
        print("Resuming from latest checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting new training...")
        trainer.train()

    trainer.save_model("/content/drive/MyDrive/qlora/complete_checkpoint")
    trainer.model.save_pretrained("/content/drive/MyDrive/qlora/final_model")

if __name__ == "__main__":
    freeze_support()
    main()