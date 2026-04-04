# ==========================================================
# 1. Bibliotheken importieren
# ==========================================================

import pandas as pd  # Für CSV-Dateien und Datenanalyse
import re  # Für Textbereinigung mit regulären Ausdrücken
import nltk  # NLP-Bibliothek
from nltk.corpus import stopwords  # Häufige irrelevante Wörter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Vektorisierung
from sklearn.decomposition import LatentDirichletAllocation  # LDA Themenmodell
from sklearn.decomposition import TruncatedSVD  # LSA Themenmodell

# Stopwords laden (nur beim ersten Start notwendig)
nltk.download("stopwords")

# ==========================================================
# 2. CSV-Datei robust laden
# ==========================================================

# Vollständiger Pfad zu deiner Datei
file_path = r"C:\Users\hawk1\Dropbox\Uni\02 Projekt Data Analysis\Phyton Code\whatsapp_reviews.csv"

# CSV robust laden
df = pd.read_csv(
    file_path,  # Dateipfad
    encoding="utf-8",  # Richtige Zeichenkodierung
    sep=",",  # Spaltentrenner
    quotechar='"',  # Texte mit Kommas korrekt lesen
    engine="python",  # Toleranter CSV-Parser
    on_bad_lines="skip"  # Fehlerhafte Zeilen überspringen
)

# Kontrollausgabe
print("Datensatz erfolgreich geladen.")
print("Anzahl Zeilen und Spalten:", df.shape)
print(df.head())

# ==========================================================
# 3. Überblick über alle Reviews
# ==========================================================

# Alle Review-Texte auswählen
all_reviews = df["text"]

# Erste 10 Reviews anzeigen
print("\n==============================")
print("ÜBERBLICK ÜBER ALLE REVIEWS")
print("==============================")
print(all_reviews.head(10))

# Anzahl aller Reviews anzeigen
print("\nGesamtanzahl Reviews:", len(all_reviews))

# ==========================================================
# 3a. Nur negative Bewertungen analysieren
# ==========================================================

# Reviews mit 1 oder 2 Sternen auswählen
negative_reviews = df[df["rating"] <= 2]

# Textspalte extrahieren
texts = negative_reviews["text"]

# Kontrollausgabe
print("\n==============================")
print("NEGATIVE REVIEWS")
print("==============================")
print("Anzahl negativer Reviews:", len(texts))
print(texts.head(10))

# ==========================================================
# 4. Texte bereinigen
# ==========================================================

# Englische Stopwords laden
stop_words = set(stopwords.words("english"))

# Funktion zur Textbereinigung
def clean_text(text):
    text = str(text)  # Sicherstellen, dass es Text ist
    text = text.lower()  # Alles klein schreiben
    
    # URLs entfernen
    text = re.sub(r"http\S+", "", text)
    
    # Sonderzeichen, Zahlen und Emojis entfernen
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Wörter trennen
    words = text.split()
    
    # Stopwords entfernen
    words = [word for word in words if word not in stop_words]
    
    # Wieder zu einem String zusammensetzen
    return " ".join(words)

# Bereinigung auf alle Texte anwenden
cleaned_texts = texts.apply(clean_text)

# Erste bereinigte Texte anzeigen
print("\nBeispiel bereinigter Texte:")
print(cleaned_texts.head())

# ==========================================================
# 5. Vektorisierung mit 2 Methoden
# ==========================================================

# ---------- Methode 1: Bag of Words ----------
count_vectorizer = CountVectorizer(max_features=1000)
X_count = count_vectorizer.fit_transform(cleaned_texts)

# ---------- Methode 2: TF-IDF ----------
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)

# ==========================================================
# 6. Themenextraktion mit 2 semantischen Methoden
# ==========================================================

# ---------- Methode 1: LDA ----------
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_count)

# ---------- Methode 2: LSA ----------
lsa = TruncatedSVD(n_components=5, random_state=42)
lsa.fit(X_tfidf)

# ==========================================================
# 7. Funktion zur Themenanzeige
# ==========================================================

def print_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nThema {topic_idx + 1}:")
        
        # Wichtigste Wörter pro Thema ausgeben
        top_words = [feature_names[i] for i in topic.argsort()[-no_top_words:]]
        print(top_words)

# ==========================================================
# 8. Ergebnisse ausgeben
# ==========================================================

print("\n==============================")
print("LDA THEMEN")
print("==============================")
print_topics(lda, count_vectorizer.get_feature_names_out())

print("\n==============================")
print("LSA THEMEN")
print("==============================")
print_topics(lsa, tfidf_vectorizer.get_feature_names_out())


