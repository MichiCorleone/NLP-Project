# =========================
# 1. Bibliotheken importieren
# =========================

import pandas as pd  # Für Datenverarbeitung (CSV laden etc.)
import re  # Für Textbereinigung mit regulären Ausdrücken
import nltk  # Für NLP-Tools (Stopwords etc.)
from nltk.corpus import stopwords  # Häufige Wörter entfernen
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Vektorisierung
from sklearn.decomposition import LatentDirichletAllocation  # LDA (Themenmodell)
from sklearn.decomposition import NMF  # Alternative Themenanalyse
import spacy  # Für fortgeschrittene Sprachverarbeitung

# Stopwords herunterladen (nur beim ersten Mal nötig)
nltk.download('stopwords')

# Spacy Modell laden
nlp = spacy.load("en_core_web_sm")

# =========================
# 2. CSV-Datei laden
# =========================

# Pfad zu deiner CSV-Datei (anpassen!)

file_path = r"C:\Users\hawk1\Dropbox\Uni\02 Projekt Data Analysis\Phyton Code\whatsapp_reviews.csv"

# CSV laden
df = pd.read_csv(file_path)

# Annahme: Die Textspalte heißt "text" (falls anders, anpassen!)
texts = df["text"]

#=========================
# 3. Textbereinigung
# =========================

# Stopwords laden
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Sicherstellen, dass der Text ein String ist
    text = str(text)
    
    # Alles klein schreiben
    text = text.lower()
    
    # URLs entfernen
    text = re.sub(r"http\S+", "", text)
    
    # Sonderzeichen und Zahlen entfernen
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenisierung (Wörter zerlegen)
    words = text.split()
    
    # Stopwords entfernen
    words = [word for word in words if word not in stop_words]
    
    # Wieder zusammenfügen
    cleaned_text = " ".join(words)
    
    return cleaned_text

# Bereinigung auf alle Texte anwenden
cleaned_text = texts.apply(clean_text)

# =========================
# 4. Vektorisierung (2 Methoden)
# =========================

# ---- Methode 1: Bag of Words ----
count_vectorizer = CountVectorizer(max_features=1000)  # Max 1000 Wörter
X_count = count_vectorizer.fit_transform(cleaned_text)  # Transformation

# ---- Methode 2: TF-IDF ----
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_text)

# =========================
# 5. Themenextraktion (2 Methoden)
# =========================

# ---- Methode 1: LDA (Latent Dirichlet Allocation) ----
lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 Themen
lda.fit(X_count)  # Modell trainieren

# ---- Methode 2: NMF (Non-negative Matrix Factorization) ----
nmf = NMF(n_components=5, random_state=42)
nmf.fit(X_tfidf)

# =========================
# 6. Themen anzeigen
# =========================

def print_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nThema {topic_idx + 1}:")
        # Top Wörter anzeigen
        print([feature_names[i] for i in topic.argsort()[-no_top_words:]])

# LDA Themen anzeigen
print("\n=== LDA Themen ===")
print_topics(lda, count_vectorizer.get_feature_names_out())

# NMF Themen anzeigen
print("\n=== NMF Themen ===")
print_topics(nmf, tfidf_vectorizer.get_feature_names_out())

# =========================
# 7. Kurzer Vergleich der Vektorisierung
# =========================

print("\n=== Vergleich Vektorisierung ===")
print("Bag of Words zählt nur Häufigkeit von Wörtern.")
print("TF-IDF gewichtet wichtige Wörter höher und häufige weniger wichtig.")

# =========================
# 8. Kurze Diskussion der Ergebnisse
# =========================

print("\n=== Diskussion ===")
print("""
- LDA erkennt Themen basierend auf Wortverteilungen (probabilistisch).
- NMF liefert oft klarere, interpretierbarere Themen.
- TF-IDF verbessert oft die Qualität der Themen gegenüber Bag of Words.
- Ergebnisse hängen stark von der Textqualität und Datenmenge ab.
""")