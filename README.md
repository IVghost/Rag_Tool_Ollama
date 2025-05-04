==================================================
     Assistant RAG Local avec Ollama (Windows/WSL)
==================================================
Application pour interroger des modèles IA locaux (Llama3, DeepSeek, Mistral) 
avec analyse de documents PDF/Word/Excel.

■■■ Prérequis ■■■
- Windows 10/11 avec WSL 2 activé
- 8 Go de RAM minimum
- Python 3.9+ installé https://www.python.org/downloads/release/python-3120/

■■■ Installation ■■■

Pré-config : Activer Hyper-V , Virtual machine plateforme) et sous-système windows pour linux dans :

Paramètres / Système / Fonctionnalités facultatives / plus de fonctionnalités Windows

1. Activer WSL (PowerShell Admin) :
----------------------------------
wsl --install
wsl --set-default-version 2

2. Installer Ollama dans WSL :
------------------------------
Dans le terminal Ubuntu/WSL :
curl -fsSL https://ollama.com/install.sh | sh

3. Dépendances Python (PowerShell/Windows) :
------------------------------------
dans ce dossier ; clic droit (dans le vide); ouvrir terminal puis : 

python -m venv venv
.\venv\Scripts\activate

python.exe -m pip install --upgrade pip

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt


- Si vous avez un GPU Nvidia : 

pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

- Si vous n'avez pas de GPU Nvidia :

pip install torch==2.5.1+cpu


■■■ Configuration Réseau ■■■
(PowerShell Admin - À faire une fois)
-------------------------------------
wsl --shutdown
wsl
sudo apt update && sudo apt install net-tools
sudo sysctl -w net.ipv4.conf.all.route_localnet=1

■■■ Utilisation ■■■

1. Démarrer Ollama (dans WSL) :
-------------------------------
ollama serve

2. Installer un modèle (nouveau terminal WSL) :
----------------------------------------------
ollama pull llama3.2 ou autres LLM à voir sur ollama.com

3. Lancer l'application (PowerShell) :
-------------------------------------
.\venv\Scripts\activate
python Rag_tool_Ivghost.py

ou

Double-clic sur le fichier start.bat 

■■■ Dépannage ■■■

► Problème : "Erreur de connexion à Ollama"
Solution :
1. Vérifier que 'ollama serve' tourne dans WSL
2. Exécuter dans PowerShell Admin :
   New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow

► Problème : "Aucun modèle détecté"
Solution :
1. Dans WSL : ollama list
2. Rafraîchir avec le bouton 🔄 dans l'interface
3. Redémarrer WSL : wsl --shutdown

■■■ Notes ■■■
- Formats supportés : .pdf, .docx, .xlsx, .txt
- Taille max fichier : 50 Mo
- Modèles recommandés : 
  * deepseek-r1 (meilleure performance)
  * llama3.1 (équilibre vitesse/précision)
  * mistral (léger)
  * llama3.2 (très léger, moins performant)
