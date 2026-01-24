"""
TinyLlama LoRA Fine-tuning Script for Clara
============================================

Prerequisites:
    pip install transformers peft accelerate bitsandbytes trl datasets

Usage:
    python train_lora.py

This will:
1. Load TinyLlama-1.1B-Chat
2. Apply LoRA adapters
3. Train on lora_ft_template.jsonl
4. Save the adapter to ./clara_lora_adapter/
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# === Configuration ===
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = Path(__file__).parent / "lora_ft_template.jsonl"
OUTPUT_DIR = Path(__file__).parent / "clara_lora_adapter"
MAX_SEQ_LENGTH = 512

# LoRA Configuration
LORA_R = 16          # Rank (higher = more capacity, more compute)
LORA_ALPHA = 32      # Scaling factor
LORA_DROPOUT = 0.05  # Dropout for regularization
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention layers


def load_dataset():
    """Load and parse the JSONL training data"""
    examples = []
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            try:
                data = json.loads(line)
                if 'text' in data:
                    examples.append(data)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def main():
    print("=" * 50)
    print("Clara LoRA Fine-tuning")
    print("=" * 50)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Training will be slow.")
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset()
    
    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with 4-bit quantization for memory efficiency
    print("\n[3/5] Loading model...")
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
    
    # Configure LoRA
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    
    # Create trainer
    print("\n[5/5] Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    # Train!
    trainer.train()
    
    # Save the adapter
    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print("\nTo use in Modal, upload this folder and load with:")
    print("  from peft import PeftModel")
    print("  model = PeftModel.from_pretrained(base_model, 'clara_lora_adapter')")
    print("=" * 50)


if __name__ == "__main__":
    main()
