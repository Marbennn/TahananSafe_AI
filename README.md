# TahananSafe AI - Incident Report Analysis System

An AI-powered system for analyzing domestic violence and abuse incident reports with multilingual support and comprehensive risk assessment.

## 🎯 Features

- **Multilingual Analysis**: English, Tagalog, Ilocano, Pangasinan, and Mixed Language
- **Abuse Classification**: Physical, Sexual, Psychological, Economic, Elder Abuse, Neglect / Acts of Omission
- **Risk Assessment**: Risk percentage (0–100%), risk level (Low/Medium/High/Critical), priority (P1/P2/P3)
- **Context Detection**: Children involved, weapon mentioned, AI confidence score
- **Fine-tuned Model**: Qwen/Qwen2.5-0.5B-Instruct with LoRA (fits 4GB GPU)

## 📁 Project Structure

```
TahananSafe_AI/
├── datasets/
│   ├── Main_Dataset.csv       # Legitimate incident reports (CSV)
│   ├── Negative_Dataset.csv   # False / irrelevant reports (CSV)
│   └── processed/             # Processed training data (from data_preparation.py)
├── models/
│   └── fine_tuned/            # Fine-tuned LoRA adapter output
├── training/
│   ├── train.py               # Training script
│   ├── data_preparation.py    # Dataset preparation (CSV or JSON/JSONL)
│   └── config.yaml            # Training and model config
├── inference/
│   ├── api.py                 # FastAPI inference server
│   ├── analyzer.py            # Core analysis logic
│   └── language_detector.py   # Language detection
├── utils/
│   ├── risk_scorer.py         # Risk percentage and level
│   └── validators.py          # Data validation
├── test_analyzer.py           # Quick test script
├── requirements.txt
├── README.md
├── SETUP.md                   # Detailed setup guide
└── QUICKSTART.md              # Short quick start
```

## 🚀 Quick Start

### 1. Virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

### 2. Prepare data and train

Datasets can be **CSV** or **JSON/JSONL**:

- **CSV**: Put `Main_Dataset.csv` and `Negative_Dataset.csv` in `datasets/`.  
  Columns: `Incident_Type`, `Incident_Description`, `Language`, `Risk_Level`, `Incident_Risk_Percentage`, `Priority_Level`, `Children_Involved`, `Weapon_Mentioned`, `AI_Confidence_Score`
- **JSON/JSONL**: Put files in `datasets/main_dataset/` and `datasets/negative_dataset/` (see SETUP.md for schema).

Then:

```bash
python training/data_preparation.py
python training/train.py
```

**Note:** Use the **activated .venv** when running training to avoid `accelerate`/`transformers` version mismatches. Training is set for a **4GB GPU** (Qwen2.5-0.5B-Instruct, batch size 1, fp16).

### 3. Run API

```bash
python inference/api.py
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

### 4. Test analyzer

```bash
python test_analyzer.py
```

## 📊 Dataset Format (CSV)

CSV files must have a header and these columns:

| Column | Description |
|--------|-------------|
| Incident_Type | e.g. Physical Abuse, Sexual Abuse, None / Invalid |
| Incident_Description | Full text of the report |
| Language | English, Tagalog, Ilocano, Pangasinan, Mixed Language |
| Risk_Level | Low, Medium, High, Critical |
| Incident_Risk_Percentage | 0–100 |
| Priority_Level | P1, P2, or P3 |
| Children_Involved | Yes / No |
| Weapon_Mentioned | Yes / No |
| AI_Confidence_Score | 0–100 |

Multi-type labels (e.g. `Physical + Psychological Abuse`) are supported in the CSV; the model learns from them as single labels.

## 🔧 Configuration

- **Model and training:** `training/config.yaml`  
  - `model.base_model`, `model.use_4bit`, `model.device_map`  
  - `training.per_device_train_batch_size`, `training.fp16`, etc.  
  - `dataset.main_dataset_path`, `dataset.negative_dataset_path` (file or directory)
- **Environment (optional):** Copy `.env.example` to `.env` and set `MODEL_PATH`, `BASE_MODEL`, `PORT`, `HF_TOKEN` if needed.

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /analyze | Analyze an incident report (body: incident_description, optional witness_name, witness_relationship, photo_urls) |
| GET | /health | Health and model-loaded status |
| GET | /model/info | Model path and device info |
| GET | /docs | Swagger UI |

## 🤖 Model Details

- **Base model:** Qwen/Qwen2.5-0.5B-Instruct
- **Fine-tuning:** PEFT (LoRA); adapter saved under `models/fine_tuned/`
- **Task:** Causal LM fine-tuned for structured incident analysis (type, language, risk, priority, children, weapon, confidence)
- **Hardware:** Tuned for 4GB GPU; can run on CPU with config changes (see SETUP.md).

## 📚 Documentation

- **SETUP.md** – Full setup, prerequisites, dataset formats, troubleshooting  
- **QUICKSTART.md** – Minimal steps to install, train, and run the API  

## License & Support

Use and modify as needed for your project. For errors, check logs and SETUP.md troubleshooting section.
