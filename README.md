# TahananSafe AI

AI-powered incident report analyzer for barangay abuse reporting and escalation support.

The system analyzes a report and returns:
- `incident_type`
- `language`
- `risk_level`
- `risk_percentage`
- `priority_level`
- `children_involved`
- `weapon_mentioned`
- `confidence_score`

## Incident Categories

Core abuse categories:
- `Physical Abuse`
- `Sexual Abuse`
- `Psychological Abuse`
- `Economic Abuse`
- `Elder Abuse`
- `Neglect / Acts of Omission`

Negative/non-abuse categories:
- `None / Invalid`
- `None / False Report`

## Latest Analyzer Logic Updates

Recent updates in `inference/analyzer.py` and `utils/risk_scorer.py`:
- Per-type pattern alignment (each abuse type has its own context/action pattern checks)
- Negation-aware physical detection (`hindi sinasaktan`, `doesn't hit`) to avoid false Physical Abuse
- Elder-victim prioritization (`elderly/grandmother/lola` + abuse context => `Elder Abuse`)
- Expanded neglect patterns:
  - food deprivation (`hindi binibigyan ng pagkain`, `not given food`)
  - medicine deprivation (`hindi binigyan ng gamot`, `not given medicine`)
- Non-human/inanimate confusion handling (object/animal actor and non-human victim blocks)
- Language arbitration to prevent false `English` on clearly Tagalog text

## Repository Structure

```text
TahananSafe_AI/
  datasets/
    Main_Dataset.csv
    Negative_Dataset.csv
    Ambiguous_Pairs.csv
  inference/
    analyzer.py
    api.py
    language_detector.py
  training/
    config.yaml
    data_preparation.py
    train.py
    evaluate_analyzer.py
    fit_confidence_calibrator.py
  utils/
    risk_scorer.py
    validators.py
    case_retriever.py
  test_analyzer.py
  requirements.txt
  .env.example
  QUICKSTART.md
  SETUP.md
```

## Team Setup (Windows, First Time)

1. Clone and enter project:

```powershell
git clone <your-repo-url>
cd TahananSafe_AI
```

2. Create and activate virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Create local environment file:

```powershell
Copy-Item .env.example .env
```

5. Verify you are using the project interpreter (very important):

```powershell
python -c "import sys,transformers,peft,accelerate; print(sys.executable); print(transformers.__version__, peft.__version__, accelerate.__version__)"
```

Expected executable path must end with:
`...\TahananSafe_AI\.venv\Scripts\python.exe`

## Daily Workflow (After Pull)

```powershell
git pull
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the System

### 1. CLI Analyzer

```powershell
python test_analyzer.py
```

### 2. API Server

```powershell
python inference/api.py
```

API endpoints:
- Swagger docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Analyze: `POST http://localhost:8000/analyze`

### 3. Sample API Call

```powershell
curl -X POST "http://localhost:8000/analyze" `
  -H "Content-Type: application/json" `
  -d "{\"incident_description\":\"The husband dragged the victim by the hair and she had broken bones.\"}"
```

## Training and Evaluation

### Prepare data

```powershell
python -c "from training.data_preparation import DataPreparator; DataPreparator('config_retrain.yaml').prepare_datasets()"
```

### Train

```powershell
python training/run_retrain.py --config config_retrain.yaml
```

### Evaluate

```powershell
python training/evaluate_analyzer.py --output training/evaluation_report.json
```

### Fit confidence calibrator

```powershell
python training/fit_confidence_calibrator.py --main-csv datasets/Main_Dataset.csv --negative-csv datasets/Negative_Dataset.csv --load-model --output models/confidence_calibrator.json
```

### Full smoke test (CLI)

```powershell
python test_analyzer.py
```

Recommended quick manual checks after retrain:
- clear physical abuse
- clear psychological abuse
- clear economic abuse
- clear sexual abuse
- clear elder abuse
- clear neglect
- explicit non-abuse/conflict-only report
- nonsense report with inanimate/non-human actor

## `.env` Notes

Start from `.env.example`.

Common keys:
- `MODEL_PATH=./models/fine_tuned`
- `MAIN_DATASET_PATH=./datasets/Main_Dataset.csv`
- `NEGATIVE_DATASET_PATH=./datasets/Negative_Dataset.csv`
- `ENABLE_CASE_RETRIEVAL=true`
- `USE_MODEL_RISK_PERCENTAGE=false`

## Most Common Issues

1. Wrong Python interpreter (global Python instead of `.venv`)
- Symptom: model load errors or old package versions.
- Fix: activate `.venv` and re-run using `python` from `.venv`.

2. `alora_invocation_tokens` error
- Cause: outdated `peft`/`transformers` in current interpreter.
- Fix:

```powershell
python -m pip install -U transformers peft accelerate
```

3. Activation opens Notepad
- Use PowerShell command exactly:

```powershell
& .\.venv\Scripts\Activate.ps1
```

4. Red `.venv` files in VS Code Git view
- This is Git tracking state, not runtime failure.
- See `SETUP.md` for safe cleanup steps.

## Team Recommendation

For consistency, all members should:
- use the same Python major/minor version
- always run from project root
- always use `.venv` interpreter
- run `pip install -r requirements.txt` after each `git pull`

For full details and troubleshooting, see:
- [QUICKSTART.md](./QUICKSTART.md)
- [SETUP.md](./SETUP.md)
