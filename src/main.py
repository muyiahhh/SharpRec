import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training
)

# 导入自定义模块
from data_moudle import SFTDataset, DataCollatorForSAFT
from finetune import SAFTTrainer

def main():
    parser = argparse.ArgumentParser(description="Run SAFT (SAM + LoRA) Finetuning for Llama 2")
    
    # 路径与模型参数
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/WeaveRec/traindata/movie.jsonl", help="Path to jsonl dataset")
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/codebase/WeaveRec/models/meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output_dir", type=str, default="./saft_output")
    
    # 训练超参数 (严格对齐 Baseline)
    parser.add_argument("--epochs", type=int, default=2, help="Aligned with Baseline: 2 epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--max_length", type=int, default=1200, help="Aligned with Baseline: 1200 cutoff")
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    
    # === 新增/修改参数 ===
    parser.add_argument("--use_sam", action="store_true", help="If set, use SAM optimizer. If not, use standard AdamW.")
    parser.add_argument("--rho", type=float, default=0.05, help="SAM perturbation magnitude (only used if --use_sam is set)")
    # ===================
    
    parser.add_argument("--use_int4", action="store_true", help="Use 4-bit quantization (QLoRA)")
    
    args = parser.parse_args()

    # --- 关键修改：动态计算梯度累积步数 ---
    # 目标：保持总 Effective Batch Size 为 32 (对齐 Baseline: 2 GPUs * 2 Batch * 8 Accum)
    target_global_batch_size = 4
    
    # 获取当前运行的显卡总数 (World Size)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 计算累积步数
    # 公式：Global_BS = GPUs * Per_Device_BS * Accum_Steps
    accum_steps = target_global_batch_size // (world_size * args.batch_size)
    accum_steps = max(1, accum_steps) # 至少为 1
    
    print(f"Automatic Configuration -> GPUs: {world_size} | Batch Per GPU: {args.batch_size} | Accumulation: {accum_steps}")
    print(f"Total Effective Batch Size: {world_size * args.batch_size * accum_steps} (Target: {target_global_batch_size})")
    # -------------------------------------

    # 1. 加载 Tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 2. 准备数据
    print("Preparing dataset...")
    train_dataset = SFTDataset(args.data_path, tokenizer, args.max_length)
    data_collator = DataCollatorForSAFT(tokenizer)
    
    # 3. 配置量化
    bnb_config = None
    if args.use_int4:
        print("Using 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False
        )

    # --- 关键修改：自动处理设备映射 (Device Map) ---
    device_map = None
    if args.use_int4:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    elif world_size > 1:
        device_map = None
    else:
        device_map = "auto"

    print(f"Loading model with device_map: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.float16
    )
    model.config.use_cache = False 
    
    if args.use_int4:
        model = prepare_model_for_kbit_training(model)

    # 5. 配置 LoRA
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,               
        lora_alpha=32,      
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"] 
    )
    
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()

    # 6. 配置 TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        # gradient_accumulation_steps=accum_steps,
        gradient_accumulation_steps=1, 
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        

        fp16=False,
        bf16=True, 
        warmup_ratio=0.05,
        lr_scheduler_type="cosine", 
        
        optim="adamw_torch",
        
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False 
    )

    # 7. 初始化自定义 Trainer
    # === 传递 use_sam 参数 ===
    trainer = SAFTTrainer(
        use_sam=args.use_sam,  # <--- 新增
        rho=args.rho,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 8. 开始训练
    mode_str = "SAM + LoRA" if args.use_sam else "Standard LoRA (SFT)"
    print(f"Starting Training Mode: {mode_str}")
    
    trainer.train()
    
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()