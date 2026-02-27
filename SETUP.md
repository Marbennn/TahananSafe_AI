# Setup Guide for TahananSafe AI

This guide walks you through installing, configuring, and running the TahananSafe AI incident report analysis system.

## Prerequisites

- **Python 3.8+** (3.10 recommended)
- **NVIDIA GPU with 4GB+ VRAM** for training (optional: CPU-only possible with config changes)
- **Hugging Face account** (for downloading the base model; no special access needed for Qwen2.5-0.5B-Instruct)
- **Git** (optional, for cloning)

## 1. Project directory

```bash
cd C:\Users\Xeium\xeium_files\TahananSafe_AI
```

## 2. Virtual environment (strongly recommended)

Using a dedicated venv avoids version conflicts between `transformers`, `accelerate`, and `peft`.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should show `(.venv)` when active.

## 3. Install dependencies

With the venv activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For **GPU training** with CUDA 12.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Optional: install `langdetect` if not already in requirements (needed for inference):

```bash
pip install langdetect
```

## 4. Hugging Face (optional)

Required only if you use a gated or private model. For the default Qwen2.5-0.5B-Instruct, download works without login. To avoid rate limits or use a different model:

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your token when prompted.

## 5. Datasets

Training supports **CSV** or **JSON/JSONL** inputs.

### Option A: CSV (recommended)

Place two files in the `datasets/` folder:

- **Main_Dataset.csv** – Legitimate incident reports (all abuse types).
- **Negative_Dataset.csv** – Non-abuse / false / irrelevant reports.

**Required columns:**

| Column | Example values |
|--------|----------------|
| Incident_Type | Physical Abuse, Sexual Abuse, Psychological Abuse, Economic Abuse, Elder Abuse, Neglect / Acts of Omission, None / Invalid |
| Incident_Description | Full text of the report |
| Language | English, Tagalog, Ilocano, Pangasinan, Mixed Language |
| Risk_Level | Low, Medium, High, Critical |
| Incident_Risk_Percentage | 0–100 (number) |
| Priority_Level | P1, P2, P3 |
| Children_Involved | Yes, No |
| Weapon_Mentioned | Yes, No |
| AI_Confidence_Score | 0–100 (number) |

You can use multi-type labels in `Incident_Type` (e.g. `Physical + Psychological Abuse`). The pipeline uses the same column for training.

Default paths in `training/config.yaml`:

- `main_dataset_path: "./datasets/Main_Dataset.csv"`
- `negative_dataset_path: "./datasets/Negative_Dataset.csv"`

### Option B: JSON / JSONL

- Put files in `datasets/main_dataset/` and `datasets/negative_dataset/`.
- Each record should have keys: `incident_description`, `incident_type`, `language`, `risk_level`, `risk_percentage`, `priority_level`, `children_involved`, `weapon_mentioned`, `confidence_score`.
- In `config.yaml`, set `main_dataset_path` and `negative_dataset_path` to these directory paths.

## 6. Prepare training data

With the venv activated and from the project root:

```bash
python training/data_preparation.py
```

This will:

- Load from the paths in `config.yaml` (CSV or JSON/JSONL).
- Train/validation/test split (default 80/10/10).
- Save processed data under `datasets/processed/`.

## 7. Train the model

**Important:** Run training **inside the same .venv** where you installed dependencies. Using the system Python can cause `Accelerator.__init__() got an unexpected keyword argument 'dispatch_batches'` or similar errors.

```bash
python training/train.py
```

Default setup (in `config.yaml`):

- **Model:** Qwen/Qwen2.5-0.5B-Instruct  
- **4GB GPU:** batch size 1, gradient accumulation 4, fp16  
- **Output:** LoRA adapter and tokenizer in `models/fine_tuned/`

**Approximate time:** Roughly 2–10 hours depending on dataset size and hardware.

To train on **CPU only** (e.g. no GPU or to avoid OOM):

- In `config.yaml`: set `model.device_map` to `"cpu"`, `training.fp16` to `false`, and optionally reduce batch size / increase gradient accumulation.

## 8. Test the analyzer

```bash
python test_analyzer.py
```

This runs sample incidents through the analyzer (fine-tuned model if present, plus rule-based fallbacks).

## 9. Run the inference API

```bash
python inference/api.py
```

Or with uvicorn:

```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
```

- **API base:** http://localhost:8000  
- **Interactive docs:** http://localhost:8000/docs  
- **Health:** http://localhost:8000/health  

### Example: analyze an incident

**POST** `/analyze` with JSON body:

```json
{
  "incident_description": "My neighbor was shouting and hitting their spouse. I heard loud noises and crying.",
  "witness_name": "Juan Dela Cruz",
  "witness_relationship": "Neighbor",
  "photo_urls": []
}
```

Response includes: `incident_type`, `language`, `risk_level`, `risk_percentage`, `priority_level`, `children_involved`, `weapon_mentioned`, `confidence_score`.

**cURL example:**

```bash
curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d "{\"incident_description\": \"My neighbor was shouting and hitting their spouse.\"}"
```

## Configuration reference

- **training/config.yaml**  
  - Model: `base_model`, `use_4bit`, `use_8bit`, `device_map`  
  - Training: `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `fp16`, `optim`  
  - Dataset: `main_dataset_path`, `negative_dataset_path`, `processed_path`, `max_length`
- **.env** (optional)  
  - Copy from `.env.example`. Set `MODEL_PATH`, `BASE_MODEL`, `PORT`, `HOST`, `HF_TOKEN` as needed.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langdetect'`

Install in the same environment you use to run the API or tests:

```bash
pip install langdetect
```

### `TypeError: Accelerator.__init__() got an unexpected keyword argument 'dispatch_batches'`

Your **global** Python has a newer `transformers` and an older `accelerate`. Fix by using the **project venv**:

```powershell
.\.venv\Scripts\activate
python training/train.py
```

If you prefer to fix the global environment:

```bash
python -m pip install --upgrade "accelerate>=0.34.0"
```

### CUDA out of memory during training

- In `config.yaml`: set `per_device_train_batch_size` to `1`, keep `gradient_accumulation_steps` at 4 or higher.
- Ensure you are using the **0.5B** model (not 1.5B) for 4GB GPUs.
- Or switch to CPU: set `device_map: "cpu"` and `fp16: false`.

### Fine-tuned model not loading / “Unrecognized model”

- The `models/fine_tuned/` folder should contain the **LoRA adapter** (e.g. `adapter_config.json`, `adapter_model.safetensors`), not a full model. The inference code loads the base model from Hugging Face and then loads the adapter. Ensure `config.yaml` and `.env` (if used) point to the same base model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`).

### Dataset format errors

- **CSV:** Ensure the first row is the header and column names match exactly (e.g. `Incident_Type`, `Incident_Description`, `Language`, etc.).
- **JSON/JSONL:** Ensure each record has the required keys; see “Option B” above.

### Training runs on CPU instead of GPU

- Install the CUDA build of PyTorch (see step 3).
- In `config.yaml`, keep `device_map: "auto"` and do not set `CUDA_VISIBLE_DEVICES` to empty when running training.

## Next steps

- Add more labeled incidents to `Main_Dataset.csv` and `Negative_Dataset.csv` and re-run data preparation and training.
- Tune `training/config.yaml` (epochs, learning rate, batch size) from validation behavior.
- Deploy the API behind a reverse proxy and secure endpoints for production use.
