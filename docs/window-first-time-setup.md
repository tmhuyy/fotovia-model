# First-Time Setup Guide

## Purpose

This document explains the first-time setup steps for running the project on a local machine.

This guide is useful when:
- setting up the project on Machine A for the first time
- moving the source code to Machine B after pulling from GitHub
- recreating the Python environment from scratch

---

## Important Principle

Do **not** push `.venv` to GitHub.

The virtual environment is local to each machine.  
Instead, only push:
- source code
- notebooks or scripts
- `requirements.txt`
- documentation files

Then, on a new machine, recreate the environment locally.

---

## Recommended Python Version

Use:

```text
Python 3.11
```

It is best to keep the same Python version on all machines to avoid package conflicts.

---

## Recommended Project Files

Make sure the repository includes:

```text
fotovia-model/
├─ docs/
│  └─ window-first-time-setup.md
├─ src/
├─ notebooks/
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## Files and Folders That Should Stay Local

These should not be pushed to GitHub:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
.env
outputs/
checkpoints/
runs/
```

---

## First-Time Setup Steps

### 1. Clone or pull the repository

If the project is not on the machine yet:

```powershell
git clone https://github.com/tmhuyy/fotovia-model.git
cd fotovia-model
```

If the project already exists:

```powershell
git switch feat(train-local-win) 
git pull
```

---

### 2. Create a virtual environment

In PowerShell:

```powershell
python -m venv .venv
```

If `python` does not work, try:

```powershell
py -3.11 -m venv .venv
```

---

### 3. Activate the virtual environment

In PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

When successful, the terminal should show something like:

```text 
Example: (.venv) PS D:\...\fotovia-model>
```

---

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

### 5. Install required packages

```powershell
pip install -r requirements.txt
```

This step recreates the local Python environment for the current machine.

---

### 6. Verify the environment

Run:

```powershell
python --version
pip --version
python -c "import torch; print(torch.__version__)"
```

If you want to check GPU support:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Important Notes

### Do not copy `.venv` from another machine

A virtual environment contains machine-specific paths and installed files.  
It may not work correctly if copied from Machine A to Machine B.

Always create a new `.venv` on each machine.

---

### Always update `requirements.txt` when adding new packages

If you install a new library locally, update the dependency file before pushing code.

You can do that with:

```powershell
pip freeze > requirements.txt
```

Or manually edit the file if you want to keep it clean and simple.

---

### Keep dataset path configurable

Dataset locations may be different on different machines.

Avoid relying on one fixed local path.

Example:

```python
Example: DATASET_ROOT = r"D:\datasets\fotovia"
```

Update this value when moving to another machine.

---

### GPU setup may differ between machines

Machine A and Machine B may not have the same GPU support.

Use safe device selection in code:

```python
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
```

This helps the project run on both GPU and CPU.

---

## Quick Setup Flow

Use this order on a new machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then open the notebook or run the training script.

---

## Troubleshooting

### `python is not recognized`

Possible reasons:
- Python is not installed
- Python is not added to PATH
- the terminal needs to be restarted

Check with:

```powershell
python --version
py --version
```

---

### Cannot activate `.venv`

Possible reasons:
- `.venv` was not created
- PowerShell blocks script execution

Fix:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

---

### Missing package error

Install dependencies again:

```powershell
pip install -r requirements.txt
```

---

## Summary

For first-time setup on any machine:

1. Pull the source code
2. Create a new `.venv`
3. Activate it
4. Install packages from `requirements.txt`
5. Check Python and PyTorch
6. Open the notebook or run the script

This is the clean and recommended workflow for using the project across multiple machines.
