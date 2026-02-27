# Quick Start – TahananSafe AI

Minimal steps to get the project running.

## 1. Virtual environment and install

```powershell
cd C:\Users\Xeium\xeium_files\TahananSafe_AI
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install langdetect
```

**GPU training (optional):**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 2. Datasets

Use **CSV** in `datasets/`:

- `Main_Dataset.csv` – abuse incident reports  
- `Negative_Dataset.csv` – non-abuse / irrelevant reports  

Columns: `Incident_Type`, `Incident_Description`, `Language`, `Risk_Level`, `Incident_Risk_Percentage`, `Priority_Level`, `Children_Involved`, `Weapon_Mentioned`, `AI_Confidence_Score`.

Config in `training/config.yaml` already points to these files. For JSON/JSONL, see SETUP.md.

## 3. Prepare and train

```powershell
python training/data_preparation.py
python training/train.py
```

**Note:** Always run these inside the **activated .venv** to avoid version errors. Training is set for a **4GB GPU** (Qwen2.5-0.5B-Instruct). Training time depends on dataset size (e.g. 30–90 minutes for hundreds of examples).

## 4. Test and run API

```powershell
python test_analyzer.py
python inference/api.py
```

- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

## 5. Call the API

```powershell
curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d "{\"incident_description\": \"My neighbor was hitting their spouse. I heard crying.\"}"
```

## Troubleshooting

| Issue | Action |
|--------|--------|
| `No module named 'langdetect'` | `pip install langdetect` in the same env |
| `Accelerator... dispatch_batches` | Use the project `.venv` and run `python training/train.py` from project root |
| CUDA OOM | In `config.yaml`: batch size 1, or set `device_map: "cpu"` and `fp16: false` for CPU training |

Full setup and troubleshooting: **SETUP.md**.
