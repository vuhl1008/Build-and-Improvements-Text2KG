import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import T5Tokenizer, T5ForConditionalGeneration

#Tải pretrained model từ HF về local
def download_model(config):
    model_id = config['id']
    model_cache = config['cache_dir']

    local_model_path = snapshot_download(model_id, cache_dir=model_cache)

    return local_model_path


