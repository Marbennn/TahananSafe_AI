# Quick Start - TahananSafe AI

Use this guide if you want to run the project fast on a new machine.

## 1. Clone and open project

```powershell
git clone <your-repo-url>
cd TahananSafe_AI
```

## 2. Create and activate virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
```

If activation fails, run directly with the venv interpreter:

```powershell
.\.venv\Scripts\python.exe --version
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Create `.env`

```powershell
Copy-Item .env.example .env
```

## 5. Verify interpreter and package versions

```powershell
python -c "import sys,transformers,peft,accelerate; print(sys.executable); print(transformers.__version__, peft.__version__, accelerate.__version__)"
```

Expected: `sys.executable` points to `.venv\Scripts\python.exe`.

## 6. Run CLI test

```powershell
python test_analyzer.py
```

## 7. Run API

```powershell
python inference/api.py
```

Open:
- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## 8. Send a test request

```powershell
curl -X POST "http://localhost:8000/analyze" `
  -H "Content-Type: application/json" `
  -d "{\"incident_description\":\"The victim said her husband kicked her while pregnant.\"}"
```

## 9. Team pull routine (every update)

```powershell
git pull
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick Fixes

- `alora_invocation_tokens` load error:

```powershell
python -m pip install -U transformers peft accelerate
```

- `No module named dotenv`:

```powershell
python -m pip install python-dotenv
```

- `python` points to global interpreter:
Use:

```powershell
.\.venv\Scripts\python.exe test_analyzer.py
```

For full setup/troubleshooting: [SETUP.md](./SETUP.md)
