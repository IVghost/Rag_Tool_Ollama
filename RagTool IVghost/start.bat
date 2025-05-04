@echo off
cd /d "%~dp0"
set VENV_DIR=venv
set REQUIREMENTS=requirements.txt
set SCRIPT= Rag_tool_Ivghost.py

REM Vérifier si le venv existe, sinon le créer
if not exist %VENV_DIR% (
    echo Création de l'environnement virtuel...
    python -m venv %VENV_DIR%
)

REM Activer le venv
call %VENV_DIR%\Scripts\activate

REM Vérifier et mettre à jour les dépendances si requirements.txt existe
if exist %REQUIREMENTS% (
    echo Vérification des mises à jour des dépendances...
    pip install --upgrade pip
    pip install -r %REQUIREMENTS%
)

REM Lancer le script Python
python %SCRIPT%

REM Désactiver le venv après exécution
deactivate

pause
