import gradio as gr
import pandas as pd
import requests
import json
import os
import torch
import psutil
import logging
import PyPDF2
import threading
import time
from docx import Document
from datetime import datetime

# Configuration API Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODELS_URL = "http://localhost:11434/api/tags"
LOG_FILE = "nutricoach.log"
API_KEYS = {
    "openai": "TA_CLE_OPENAI",
    "anthropic": "TA_CLE_ANTHROPIC",
    "perplexity": "TA_CLE_PERPLEXITY"
}

stop_event = threading.Event()

# Configuration du logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

print("Torch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Nombre de GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Nom du GPU:", torch.cuda.get_device_name(0))
    print("Mémoire GPU:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(i)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} Go")


def log(message):
    """Affiche les logs dans le terminal"""
    print(f"[LOG] {message}")

#Affichage connexion Ollama
def check_connection():
    """Vérifie la connexion à Ollama"""
    try:
        response = requests.get(OLLAMA_MODELS_URL, timeout=2)
        if response.status_code == 200:
            return "🟢 Connecté à Ollama"
    except Exception as e:
        return f"🔴 Ollama non disponible ({str(e)})"
    return "🔴 Ollama non disponible"

# Fonction pour récupérer les modèles disponibles
def get_available_models():
    try:
        response = requests.get(OLLAMA_MODELS_URL, timeout=2)
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            return models if models else ["Aucun modèle disponible"]
    except Exception as e:
        return [f"Erreur de connexion: {str(e)}"]
    return ["Erreur de connexion"]


def stop_operations():
    """Déclenche un signal pour arrêter toutes les opérations en cours."""
    global stop_event
    stop_event.set()  # Déclenche l'arrêt
    log(f"[{time.strftime('%H:%M:%S')}] 🛑 Demande d'arrêt reçue. Tentative de libération des ressources...")

    # Essayer de libérer le modèle si possible
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Libérer la mémoire GPU
            log(f"[{time.strftime('%H:%M:%S')}] ✅ Mémoire GPU libérée.")
        else:
            log(f"[{time.strftime('%H:%M:%S')}] ⚠️ Pas de GPU détecté, pas de mémoire à libérer.")

        # Terminer tous les threads actifs
        for thread in threading.enumerate():
            if thread.is_alive() and thread != threading.main_thread():
                log(f"[{time.strftime('%H:%M:%S')}] ⏳ Tentative d'arrêt du thread {thread.name}...")
                thread.join(timeout=1)

        log(f"[{time.strftime('%H:%M:%S')}] ✅ Tous les threads ont été arrêtés.")

    except Exception as e:
        log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur lors de l'arrêt des opérations : {e}")


# Extraction de contenu des fichiers
def extract_content(file_path):
    lines = []
    
    start_time = time.time()
    log(f"[{time.strftime('%H:%M:%S')}] 🔄 Début de l'extraction du fichier : {file_path}")

    if stop_event.is_set():
        log(f"[{time.strftime('%H:%M:%S')}] 🛑 Arrêt forcé détecté. Fin de l'extraction.")
        return "❌ Extraction annulée."

    try:
        if file_path.endswith(".pdf"):
            log(f"[{time.strftime('%H:%M:%S')}] 📄 Fichier PDF détecté, lecture en cours...")
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    if stop_event.is_set():
                        log(f"[{time.strftime('%H:%M:%S')}] 🛑 Arrêt forcé pendant la lecture du PDF.")
                        return "❌ Extraction annulée."
                    text = page.extract_text()
                    if text:
                        lines.append(text)
                    log(f"[{time.strftime('%H:%M:%S')}] ✅ Page {page_num + 1} extraite.")

        elif file_path.endswith(".docx"):
            log(f"[{time.strftime('%H:%M:%S')}] 📄 Fichier DOCX détecté, lecture en cours...")
            doc = Document(file_path)
            for idx, para in enumerate(doc.paragraphs):
                if stop_event.is_set():
                    log(f"[{time.strftime('%H:%M:%S')}] 🛑 Arrêt forcé pendant la lecture du DOCX.")
                    return "❌ Extraction annulée."
                lines.append(para.text)
                log(f"[{time.strftime('%H:%M:%S')}] ✅ Paragraphe {idx + 1} extrait.")

        elif file_path.endswith(".csv") or file_path.endswith(".xlsx"):
            log(f"[{time.strftime('%H:%M:%S')}] 📊 Fichier tabulaire détecté, lecture en cours...")
            df = load_csv_safely(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
            lines.append(df.to_string())

        else:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Type de fichier non pris en charge.")

    except Exception as e:
        log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur lors de l'extraction du fichier : {e}")

    total_time = time.time() - start_time
    log(f"[{time.strftime('%H:%M:%S')}] ✅ Extraction terminée en {total_time:.2f} secondes.\n")
    return "\n".join(lines)




# Calcul des macros manquantes
def calculate_missing_macros(protein, carbs, fats, kcal):
    try:
        p = float(protein) if protein not in [None, ""] else None
        c = float(carbs) if carbs not in [None, ""] else None
        f = float(fats) if fats not in [None, ""] else None
        k = float(kcal) if kcal not in [None, ""] else None

        if all(v is not None for v in [p, c, f]):
            return p, c, f, p*4 + c*4 + f*9

        if k:
            known_cal = sum([p*4 if p else 0, c*4 if c else 0, f*9 if f else 0])
            remaining = k - known_cal
            missing = sum(1 for v in [p, c, f] if v is None)

            if missing == 1:
                if p is None: p = remaining / 4
                elif c is None: c = remaining / 4
                else: f = remaining / 9
            elif missing == 2:
                if p is not None:
                    c = (remaining * 0.6) / 4
                    f = (remaining * 0.4) / 9
                else:
                    p = (remaining * 0.3) / 4
                    c = (remaining * 0.4) / 4
                    f = (remaining * 0.3) / 9
            elif missing == 3:
                p = (k * 0.3) / 4
                c = (k * 0.4) / 4
                f = (k * 0.3) / 9

            return round(p, 1), round(c, 1), round(f, 1), k

        raise ValueError("Au moins un paramètre doit être fourni")

    except Exception as e:
        raise ValueError(f"Erreur de calcul: {str(e)}")



def load_csv_safely(file_path):
    """Charge un fichier CSV en détectant automatiquement son format et en évitant les erreurs de parsing."""
    
    # 🔍 Étape 1 : Détecter le séparateur
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    
    possible_separators = [",", ";", "\t", "|"]
    detected_separator = max(possible_separators, key=lambda sep: first_line.count(sep))
    
    print(f"✅ Séparateur détecté : {repr(detected_separator)}")

    # 🔍 Étape 2 : Lecture du fichier CSV en évitant les erreurs
    try:
        df = pd.read_csv(
            file_path, 
            sep=detected_separator, 
            encoding="utf-8",
            engine="python",
            on_bad_lines=lambda line: print(f"⚠️ Ligne ignorée : {line}") or None  # Log des lignes corrompues
        )

    except UnicodeDecodeError:
        print("⚠️ Erreur d'encodage détectée, tentative avec ISO-8859-1...")
        df = pd.read_csv(file_path, sep=detected_separator, encoding="ISO-8859-1", engine="python")

    print(f"✅ Fichier chargé avec succès ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    
    return df


def clean_nutrition_data(json_path):
    """Charge et nettoie les données JSON."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        for key, value in item.items():
            if isinstance(value, str):
                # Nettoyage des espaces et conversion des nombres
                value = value.strip().replace(",", ".")
                try:
                    item[key] = float(value) if value else None
                except ValueError:
                    pass  # Garder les chaînes de texte inchangées

        # ✅ Remplacer les valeurs nulles par 0 ou une valeur par défaut
        item["Energie (kcal/100 g)"] = item.get("Energie (kcal/100 g)", 0) or 0
        item["Protéines (g/100 g)"] = item.get("Protéines (g/100 g)", 0) or 0
        item["Glucides (g/100 g)"] = item.get("Glucides (g/100 g)", 0) or 0
        item["Lipides (g/100 g)"] = item.get("Lipides (g/100 g)", 0) or 0
        item["Fibres alimentaires (g/100 g)"] = item.get("Fibres alimentaires (g/100 g)", 0) or 0

    return pd.DataFrame(data)


def generate_nutrition_plan(file, protein, carbs, fats, kcal, diet_type, model, meals):
    """Génère un plan nutritionnel et log chaque étape du traitement."""
    try:
        start_time = time.time()  # Début du chronomètre
        log(f"[{time.strftime('%H:%M:%S')}] 🔄 Début du traitement pour le plan nutritionnel...")

        if stop_event.is_set():
            log(f"[{time.strftime('%H:%M:%S')}] 🛑 Arrêt forcé détecté. Fin du traitement.")
            return "❌ Opération annulée."

        # Vérification du fichier
        if file is None:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Aucun fichier fourni.")
            return "❌ Veuillez télécharger un fichier nutritionnel"

        file_path = file.name if hasattr(file, "name") else None
        if not file_path:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Fichier non valide.")
            return "❌ Fichier non valide"
        if stop_event.is_set():
            log(f"[{time.strftime('%H:%M:%S')}] 🛑 Arrêt forcé détecté. Fin du traitement.")
            return "❌ Opération annulée."


        log(f"[{time.strftime('%H:%M:%S')}] 📂 Fichier reçu : {file_path}")

        # Vérification et conversion des macros
        try:
            protein = float(protein) if protein not in [None, ""] else 0
            carbs = float(carbs) if carbs not in [None, ""] else 0
            fats = float(fats) if fats not in [None, ""] else 0
            kcal = float(kcal) if kcal not in [None, ""] else (protein * 4 + carbs * 4 + fats * 9)
        except ValueError:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Valeurs nutritionnelles invalides.")
            return "❌ Valeurs nutritionnelles incorrectes"

        log(f"[{time.strftime('%H:%M:%S')}] ✅ Macros validées : {protein}g protéines, {carbs}g glucides, {fats}g lipides, {kcal} kcal.")

        # Vérification du nombre de repas
        try:
            meals = int(meals) if meals not in [None, ""] else None
            if meals is None or meals <= 0:
                raise ValueError("Nombre de repas invalide.")
        except (ValueError, TypeError):
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Nombre de repas invalide.")
            return "❌ Le nombre de repas est invalide."

        log(f"[{time.strftime('%H:%M:%S')}] ✅ Nombre de repas validé : {meals} repas/jour.")

        # Lecture du fichier et extraction des aliments
        log(f"[{time.strftime('%H:%M:%S')}] 📊 Lecture et analyse du fichier nutritionnel en cours...")
        try:
            if file_path.endswith('.csv'):
                df = load_csv_safely(file_path)
            elif file_path.endswith('.json'):
                df = clean_nutrition_data(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Format de fichier non supporté.")
                return "❌ Format de fichier non supporté"
        except Exception as e:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur lors de la lecture du fichier : {e}")
            return f"❌ Erreur lors de la lecture du fichier : {str(e)}"

        food_data = df.head(20).to_dict('records')
        log(f"[{time.strftime('%H:%M:%S')}] ✅ {len(food_data)} aliments extraits avec succès.")


        if stop_event.is_set():
            log(f"[{time.strftime('%H:%M:%S')}] 🛑 Arrêt forcé détecté. Fin du traitement.")
            return "❌ Opération annulée."

        # Construction du prompt pour l'IA
        print("📜 Construction du prompt pour l'IA...")
        prompt = (
            f"Tu es un assistant expert en nutrition. Génére un plan alimentaire sur **7 jours** avec exactement **{meals} repas par jour**, en respectant les macros ci-dessous.\n\n"

            f"⚠️ **Objectifs nutritionnels quotidiens** :\n"
            f"- **{kcal:.0f} kcal**\n"
            f"- **{protein:.1f}g protéines** ({protein / meals:.1f}g par repas)\n"
            f"- **{carbs:.1f}g glucides** ({carbs / meals:.1f}g par repas)\n"
            f"- **{fats:.1f}g lipides** ({fats / meals:.1f}g par repas)\n\n"

            f"📌 **Base de données des aliments autorisés** (utilise uniquement ces aliments) :\n"
            f"{json.dumps(food_data, indent=2)}\n\n"

            "📝 **Instructions claires** :\n"
            "- Chaque **jour doit être unique**, aucun repas ne doit être répété.\n"
            "- Chaque **aliment doit avoir sa quantité précise en grammes**.\n"
            "- Chaque **repas doit respecter les macros du jour**.\n"
            "- Ne rajoute **aucune phrase explicative** en dehors du tableau.\n\n"

            "⚠️ **IMPORTANT :** Affiche UNIQUEMENT le tableau suivant sans explications :\n\n"

            "🟢 **FORMAT DU TABLEAU ATTENDU** :\n\n"

            "| Jour | Repas       | Aliment           | Quantité (g) | Énergie (kcal) | Protéines (g) | Glucides (g) | Lipides (g) |\n"
            "|------|------------|-------------------|--------------|----------------|---------------|--------------|-------------|\n"
            "| 1    | Petit-déj  | Flocons d'avoine  | 50           | 180            | 6             | 30           | 4           |\n"
            "| 1    | Déjeuner   | Poulet grillé     | 150          | 230            | 35            | 0            | 5           |\n"
            "| 1    | Dîner      | Saumon            | 120          | 250            | 25            | 0            | 15          |\n"
            "| 2    | Petit-déj  | Yaourt nature     | 150          | 90             | 8             | 7            | 4           |\n"
            "| 2    | Déjeuner   | Riz complet       | 100          | 130            | 3             | 28           | 1           |\n"
            "| 2    | Dîner      | Poisson grillé    | 120          | 180            | 35.5          | 1.8          | 7.2         |\n"
            "...\n\n"

            "Ne modifie **pas** la structure du tableau. **Ne rajoute rien avant ou après**. Affiche **uniquement** le tableau final, sans explication."
        )



        log(f"[{time.strftime('%H:%M:%S')}] 🚀 Envoi du prompt au LLM...")
        log(f"[{time.strftime('%H:%M:%S')}] 📝 Aperçu du prompt : {prompt[:200]}... [Tronqué]")

        # Chronométrage du temps de réponse du LLM
        llm_start_time = time.time()
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False}
        )
        llm_duration = time.time() - llm_start_time

        # Vérification du statut de la réponse
        if response.status_code == 200:
            log(f"[{time.strftime('%H:%M:%S')}] ✅ Réponse reçue du LLM en {llm_duration:.2f} secondes.")
            result = response.json().get("response", "Aucune réponse obtenue")
        else:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur API : {response.status_code} - {response.text}")
            return f"❌ Erreur API : {response.status_code} - {response.text}"

        total_time = time.time() - start_time
        log(f"[{time.strftime('%H:%M:%S')}] ✅ Plan nutritionnel généré en {total_time:.2f} secondes.\n")
        return result

    except Exception as e:
        log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur inattendue : {e}")
        return f"❌ Erreur : {str(e)}"


def analyze_document(file, question, model):
    """Analyse un document avec l'Assistant Général et log chaque étape."""
    try:
        start_time = time.time()
        log(f"[{time.strftime('%H:%M:%S')}] 🔄 Début de l'analyse du document...")

        if file is None:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Aucun fichier fourni.")
            return "❌ Veuillez télécharger un fichier."

        file_path = file.name if hasattr(file, "name") else None
        if not file_path:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Fichier non valide.")
            return "❌ Fichier non valide"

        log(f"[{time.strftime('%H:%M:%S')}] 📂 Fichier reçu : {file_path}")
        log(f"[{time.strftime('%H:%M:%S')}] 📝 Question posée : {question}")

        # Extraction du contenu du fichier
        log(f"[{time.strftime('%H:%M:%S')}] 📄 Extraction du contenu en cours...")
        doc_start_time = time.time()
        extracted_text = extract_content(file_path)
        doc_duration = time.time() - doc_start_time
        log(f"[{time.strftime('%H:%M:%S')}] ✅ Contenu extrait en {doc_duration:.2f} secondes.")

        # Vérification du modèle sélectionné
        if not model or model == "Aucun modèle disponible":
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur : Aucun modèle sélectionné.")
            return "❌ Veuillez sélectionner un modèle LLM."


        if stop_event.is_set():
            log(f"[{time.strftime('%H:%M:%S')}] 🛑 Arrêt forcé détecté. Fin du traitement.")
            return "❌ Opération annulée."

        # Construction du prompt pour le LLM
        log(f"[{time.strftime('%H:%M:%S')}] 📜 Construction du prompt pour l'IA...")
        prompt = f"Document:\n{extracted_text}\n\nQuestion: {question}"

        # Envoi à l'API Ollama
        log(f"[{time.strftime('%H:%M:%S')}] 🚀 Envoi de la requête au LLM...")
        llm_start_time = time.time()
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False}
        )
        llm_duration = time.time() - llm_start_time

        # Vérification du statut de la réponse
        if response.status_code == 200:
            log(f"[{time.strftime('%H:%M:%S')}] ✅ Réponse reçue en {llm_duration:.2f} secondes.")
            result = response.json().get("response", "Aucune réponse obtenue")
        else:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur API : {response.status_code} - {response.text}")
            return f"❌ Erreur API : {response.status_code} - {response.text}"

        total_time = time.time() - start_time
        log(f"[{time.strftime('%H:%M:%S')}] ✅ Analyse du document terminée en {total_time:.2f} secondes.\n")
        return result

    except Exception as e:
        log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur inattendue : {e}")
        return f"❌ Erreur : {str(e)}"


def query_llm(prompt, model, provider, api_key):
    """Envoie le prompt au LLM choisi (Ollama en local ou API externe)."""
    log(f"[{time.strftime('%H:%M:%S')}] 🔄 Envoi du prompt au LLM ({provider})...")

    start_time = time.time()

    try:
        if provider == "Ollama (local)":
            url = OLLAMA_URL
            payload = {"model": model, "prompt": prompt, "stream": False}
            headers = {}
        elif provider == "OpenAI":
            url = "https://api.openai.com/v1/completions"
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            headers = {"Authorization": f"Bearer {api_key}"}
        elif provider == "Anthropic":
            url = "https://api.anthropic.com/v1/complete"
            payload = {
                "model": model,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 1000
            }
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json"
            }
        elif provider == "Perplexity":
            url = "https://api.perplexity.ai/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            headers = {"Authorization": f"Bearer {api_key}"}
        else:
            return "❌ Fournisseur IA non pris en charge."

        response = requests.post(url, json=payload, headers=headers)
        duration = time.time() - start_time

        if response.status_code == 200:
            log(f"[{time.strftime('%H:%M:%S')}] ✅ Réponse reçue en {duration:.2f} secondes.")
            return response.json().get("choices", [{}])[0].get("text", "Réponse vide").strip()
        else:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur API : {response.status_code} - {response.text}")
            return f"❌ Erreur API : {response.status_code} - {response.text}"

    except Exception as e:
        log(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur lors de l'interrogation de l'IA : {str(e)}")
        return f"❌ Erreur lors de l'interrogation de l'IA : {str(e)}"


# Interface Gradio
with gr.Blocks(title="👻 IvGhost RAG tool", theme=gr.themes.Soft()) as app:
    gr.Markdown("# RAG AI Tool")
    connection_status = gr.Markdown(value=check_connection(), elem_id="status")
    refresh_button = gr.Button("🔄 Vérifier connexion")
    refresh_button.click(lambda: check_connection(), outputs=connection_status)

    model_selector = gr.Dropdown(
        label="Modèle LLM (Ollama uniquement)",
        choices=get_available_models(),
        interactive=True
    )
    llm_provider = gr.Dropdown(
        label="Fournisseur IA",
        choices=["Ollama (local)", "OpenAI", "Anthropic", "Perplexity"],
        value="Ollama (local)",
        interactive=True
    )
    api_key_input = gr.Textbox(label="🔑 Clé API (si externe)", type="password", interactive=True)

    stop_button = gr.Button("🛑 Arrêter toutes les opérations")
    stop_button.click(stop_operations)

    # ✅ FIX: Move `gr.Row()` inside `gr.Blocks()`
    with gr.Row():
        model_selector = gr.Dropdown(
            label="Modèle LLM",
            choices=get_available_models(),
            interactive=True
        )
        refresh_model_button = gr.Button("🔄 Actualiser les modèles")
        refresh_model_button.click(
            lambda: gr.Dropdown.update(choices=get_available_models()),
            outputs=model_selector
        )

    with gr.Tabs():
        with gr.Tab("💬 Assistant Général"):
            gr.Markdown("## Analysez vos documents")
            with gr.Row():
                doc_input = gr.File(label="Document (PDF/CSV/Excel)")
                question_input = gr.Textbox(label="Question", lines=3)

            doc_output = gr.Textbox(label="Réponse", interactive=False)
            gr.Button("🔍 Analyser").click(
                analyze_document,
                inputs=[doc_input, question_input, model_selector],
                outputs=doc_output
            )

        with gr.Tab("🍎 Assistant Nutrition Intelligent"):
            with gr.Row():
                file_input = gr.File(label="Base de données alimentaire", file_types=[".csv", ".json", ".xlsx"])
                with gr.Column():
                    protein_input = gr.Number(label="Protéines (g/jour)", interactive=True)
                    carbs_input = gr.Number(label="Glucides (g/jour)", interactive=True)
                    fats_input = gr.Number(label="Lipides (g/jour)", interactive=True)
                    kcal_input = gr.Number(label="Calories cibles", interactive=True)
                    meals_input = gr.Number(label="Nombre de repas/jour", interactive=True, precision=0)
                    diet_type_input = gr.Dropdown(["Standard", "Végétarien", "Vegan", "Cétogène"], value="Standard", label="Type de régime")

            output = gr.Markdown()
            gr.Button("🚀 Générer le plan").click(
                generate_nutrition_plan,
                inputs=[file_input, protein_input, carbs_input, fats_input, kcal_input, diet_type_input, model_selector, meals_input],
                outputs=output
            )

if __name__ == "__main__":
    app.launch(server_port=7860)
