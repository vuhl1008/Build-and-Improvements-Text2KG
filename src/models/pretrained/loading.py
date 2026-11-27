from huggingface_hub import snapshot_download

#Tải pretrained model từ HF về local
def download_model(config):
    return snapshot_download(
        repo_id=config["id"],
        revision=config.get("revision", "main"),
        cache_dir=config.get("cache_dir"),
    )


