import torch
from trl import SFTConfig, SFTTrainer

from src.data.prepare import load_raw_splits
from src.models.pretrained.dataset import format_for_sft
from src.models.pretrained.loading import download_model
from src.models.pretrained.model import build_lora_config, load_base_model


def train(config):
    if config["model"].get("require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training configuration")
    #Model
    model_path = download_model(config["model"])
    tokenizer, model = load_base_model(model_path, config["model"].get("torch_dtype", "float16"))

    #Data
    dataset = load_raw_splits(config["dataset"]["processed_dir"]) 
    sft_dataset = {split: format_for_sft(rows) for split, rows in dataset.items()}
    train_data = sft_dataset[config["dataset"]["splits"][0]]
    validation_data = sft_dataset[config["dataset"]["splits"][1]]

    #Training Args
    training_args = SFTConfig(
        output_dir=config["training"]["output_dir"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        max_length=config["training"]["max_length"],
        logging_steps=config["training"]["logging_steps"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        report_to=[],
        bf16=False,
        assistant_only_loss=True,
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        seed=config["training"]["seed"],
    )

    #Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        peft_config=build_lora_config(config["lora"]),
        processing_class=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])
    return trainer
