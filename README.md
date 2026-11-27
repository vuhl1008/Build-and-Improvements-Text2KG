# Build-and-Improvements-Text2KG

Fine-tune Qwen2.5-1.5B-Instruct để chuyển văn bản thành các bộ ba knowledge graph theo WebNLG.

## Cài đặt và tải dữ liệu

```bash
pip install -r requirements.txt
python scripts/download_assets.py
```

Các đường dẫn tải được cấu hình trong `configs/download.yaml`. Dữ liệu JSONL gốc nằm trong `data/raw`.

## Huấn luyện

Chạy trên máy có CUDA:

```bash
python scripts/train.py
```

Thiết lập fine-tuning nằm trong `configs/fine_tuning.yaml`. Tiền xử lý WebNLG dùng chung nằm trong `src/data`; phần Qwen/LoRA nằm trong `src/models/pretrained`. Adapter và tokenizer được lưu trong `outputs/Qwen-text2kg-lora`.

## Sinh bộ ba

```bash
python scripts/predict.py "The text to convert into knowledge graph triples" --adapter outputs/Qwen-text2kg-lora
```
