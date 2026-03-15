# Setup Guide - TahananSafe AI

This guide is for team members who will pull the repo and run the analyzer/API locally.

## 1. Prerequisites

- Windows 10/11 (PowerShell) or Linux/macOS shell
- Python 3.10 recommended
- Git
- Optional GPU: NVIDIA + CUDA-compatible PyTorch

## 2. Clone the repository

```powershell
git clone <your-repo-url>
cd TahananSafe_AI
```

## 3. Create and activate virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Configure environment variables

Create local `.env` from template:

```powershell
Copy-Item .env.example .env
```

Minimum recommended values in `.env`:

```dotenv
MODEL_PATH=./models/fine_tuned
MAIN_DATASET_PATH=./datasets/Main_Dataset.csv
NEGATIVE_DATASET_PATH=./datasets/Negative_Dataset.csv
ENABLE_CASE_RETRIEVAL=true
USE_MODEL_RISK_PERCENTAGE=false
```

## 6. Verify correct interpreter (critical)

Run:

```powershell
python -c "import sys,transformers,peft,accelerate; print(sys.executable); print(transformers.__version__, peft.__version__, accelerate.__version__)"
```

`sys.executable` must be inside:
`...\TahananSafe_AI\.venv\Scripts\python.exe`

If not, use explicit interpreter:

```powershell
.\.venv\Scripts\python.exe test_analyzer.py
```

## 7. Dataset setup

Expected files in `datasets/`:
- `Main_Dataset.csv`
- `Negative_Dataset.csv`
- `Ambiguous_Pairs.csv` (optional but recommended)

Required CSV columns:
- `Incident_Type`
- `Incident_Description`
- `Language`
- `Risk_Level`
- `Incident_Risk_Percentage`
- `Priority_Level`
- `Children_Involved`
- `Weapon_Mentioned`
- `AI_Confidence_Score`

## 8. Run the analyzer

### CLI test

```powershell
python test_analyzer.py
```

### API server

```powershell
python inference/api.py
```

Endpoints:
- Health: `GET http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`
- Analyze: `POST http://localhost:8000/analyze`

## 9. Connect client app to API

### JavaScript `fetch` example

```javascript
const payload = {
  incident_description: "The husband dragged the victim by the hair and she had broken bones."
};

const res = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});

const data = await res.json();
console.log(data);
```

### Axios example

```javascript
import axios from "axios";

const { data } = await axios.post("http://localhost:8000/analyze", {
  incident_description: "Napapansin ko na madalas walang pagkain at tubig sa bahay ng mga bata."
});

console.log(data);
```

## 10. Optional training pipeline

```powershell
python -c "from training.data_preparation import DataPreparator; DataPreparator('config_retrain.yaml').prepare_datasets()"
python training/run_retrain.py --config config_retrain.yaml
python training/evaluate_analyzer.py --output training/evaluation_report.json
```

Confidence calibration:

```powershell
python training/fit_confidence_calibrator.py --main-csv datasets/Main_Dataset.csv --negative-csv datasets/Negative_Dataset.csv --load-model --output models/confidence_calibrator.json
```

## 11. CUDA check (optional)

```powershell
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
```

If CUDA is `False`, project still runs on CPU (slower).

## 12. Team pull/update routine

Every time you pull new changes:

```powershell
git pull
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

## 13. Troubleshooting

### A. `alora_invocation_tokens` model load error

Cause: old `peft`/`transformers` in current Python.

Fix:

```powershell
python -m pip install -U transformers peft accelerate
```

### B. `ModuleNotFoundError: No module named dotenv`

```powershell
python -m pip install python-dotenv
```

### C. Activation command opens Notepad

Use this exact PowerShell command:

```powershell
& .\.venv\Scripts\Activate.ps1
```

Or bypass activation and run direct:

```powershell
.\.venv\Scripts\python.exe inference/api.py
```

### D. VS Code shows many red files under `.venv`

This is usually a Git tracking issue, not runtime failure.

Safe local cleanup of staged `.venv` deletions:

```powershell
git restore --staged .venv
```

Permanent repo cleanup (maintainer does once):

```powershell
git rm -r --cached .venv
git commit -m "Stop tracking .venv"
```

`.gitignore` should include `.venv/`.

### E. API does not respond from another device

- Ensure server is running with host `0.0.0.0`.
- Check firewall allows port `8000`.
- Use machine IP, not `localhost`, from remote device.

### F. Tagalog sentence shows `Language: English`

- Stop old running process and start a fresh one.
- Verify latest pull includes language-arbitration logic.
- Retest:

```powershell
python test_analyzer.py
```

### G. Elder-neglect or child-neglect report gets blocked incorrectly

- Ensure latest code is pulled (medicine/food deprivation patterns were expanded).
- Retrain if you changed datasets:

```powershell
python -c "from training.data_preparation import DataPreparator; DataPreparator('config_retrain.yaml').prepare_datasets()"
python training/run_retrain.py --config config_retrain.yaml
```

### H. Confusing report with `doesn't hit` still classified as Physical Abuse

- Latest logic includes negation-aware physical detection.
- Restart CLI/API process and retest.

## 14. Defense-day checklist

Before demo/presentation:

1. Activate `.venv`.
2. Verify interpreter path.
3. Run `python inference/api.py`.
4. Open `/health` and `/docs`.
5. Test at least 3 incidents (low, high, critical).
6. Keep backup commands ready in terminal history.
