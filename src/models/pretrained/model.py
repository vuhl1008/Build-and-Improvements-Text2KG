import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.pretrained.prompt import SYSTEM_PROMPT


def load_base_model(model_path, torch_dtype="float16"):
    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.config.use_cache = True
    return tokenizer, model


def build_lora_config(config):
    return LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        bias="none",
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
    )


def load_adapter(base_model, adapter_path):
    return PeftModel.from_pretrained(base_model, str(adapter_path))


def generate_answer(model, tokenizer, question, max_new_tokens=100, num_beams=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(model.device)
    model.config.use_cache = True
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
