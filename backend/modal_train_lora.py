"""
Clara LoRA Fine-tuning on Modal
================================

Trains TinyLlama with LoRA on Modal's A10G GPU.
Saves the adapter to a Modal Volume for persistence.

Usage:
    modal run modal_train_lora.py
"""

import modal
from pathlib import Path

# === Modal Setup ===
APP_NAME = "clara-lora-training"

# Create a volume to persist the trained adapter
volume = modal.Volume.from_name("clara-lora-adapters", create_if_missing=True)

# Training image with all dependencies
# Use compatible versions (tested combination)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.38.2",  # Compatible with trl 0.8
        "accelerate==0.27.2",
        "bitsandbytes==0.42.0",
        "peft==0.8.2",
        "trl==0.8.1",  # Works with transformers 4.38
        "datasets==2.17.0",
        "huggingface-hub==0.21.3",
        "rich",  # Required by trl
    )
)

app = modal.App(APP_NAME)

# Add training data to image
image = image.add_local_file(
    Path(__file__).parent / "clara_training_data.jsonl",
    remote_path="/root/clara_training_data.jsonl"
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,  # 1 hour max
    volumes={"/output": volume},
)
def train_lora():
    """Run LoRA fine-tuning on TinyLlama"""
    import json
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    
    print("=" * 60)
    print("Clara LoRA Fine-tuning")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # === Configuration ===
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = "/output/clara_lora_adapter"
    MAX_SEQ_LENGTH = 512
    
    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # === Load Dataset ===
    print("\n[1/5] Loading dataset...")
    examples = []
    with open("/root/clara_training_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    
    dataset = Dataset.from_list(examples)
    print(f"Loaded {len(examples)} training examples")
    
    # === Load Tokenizer ===
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # === Load Model ===
    print("\n[3/5] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    
    # === Configure LoRA ===
    print("\n[4/5] Configuring LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # === Training Arguments (trl 0.7.x API) ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    )
    
    # === Train! ===
    print("\n[5/5] Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    # Train
    trainer.train()
    
    # Save final adapter
    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Commit volume changes
    volume.commit()
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"Adapter saved to Modal Volume: clara-lora-adapters")
    print(f"Path: {OUTPUT_DIR}")
    print("=" * 60)
    
    return {"status": "success", "examples": len(examples), "output_dir": OUTPUT_DIR}


@app.function(image=image, volumes={"/output": volume})
def list_adapters():
    """List saved adapters in the volume"""
    import os
    
    adapter_dir = "/output/clara_lora_adapter"
    if os.path.exists(adapter_dir):
        files = os.listdir(adapter_dir)
        print(f"Adapter files: {files}")
        return files
    else:
        print("No adapters found yet")
        return []


@app.local_entrypoint()
def main():
    """Run training"""
    print("Starting Clara LoRA training on Modal...")
    result = train_lora.remote()
    print(f"Training result: {result}")
