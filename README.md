# NLP-Projekt: Analyse von WhatsApp User Reviews

**Kurs:** Projekt Data Analysis (DLBDSEDA02_D)
**Aufgabe:** NLP-Techniken zur Analyse einer Sammlung von Beschwerdetexten

Dieses Repository enthält den vollständigen Code, den verwendeten Datensatz sowie die
Dokumentation der drei Portfolio-Phasen (Konzept, Erarbeitung, Finalisierung).

## Ziel

Aus einer großen Sammlung unstrukturierter Beschwerdetexte sollen mithilfe von
NLP-Techniken automatisch die häufigsten Themen extrahiert und visualisiert werden,
damit Entscheidungsträger:innen rasch einen Überblick erhalten.

Als Beispieldatensatz dient *WhatsApp User Reviews*
([Kaggle](https://www.kaggle.com/datasets/sonalshinde123/whatsapp-user-reviews-dataset/data)).
Die Pipeline ist generisch aufgebaut und lässt sich ohne strukturelle Änderungen auf
andere Beschwerdedatensätze übertragen (z. B. Stadtverwaltung, Kundenbeschwerden,
Produktbewertungen).

## Pipeline-Überblick

1. **Daten laden** (`pandas`) – robustes CSV-Parsing mit Toleranz für fehlerhafte Zeilen
2. **Filterung** – nur negative Reviews (≤ 2 Sterne) werden analysiert
3. **Textbereinigung** – Kleinschreibung, Entfernen von URLs, Sonderzeichen, Zahlen
   und englischen Stoppwörtern (NLTK), anschließend Tokenisierung
4. **Vektorisierung** – Bag of Words *und* TF-IDF (scikit-learn)
5. **Themenextraktion** – LDA (auf BoW) *und* LSA / TruncatedSVD (auf TF-IDF),
   jeweils 5 Themen
6. **Visualisierung** – vergleichende Balkendiagramme (LDA vs. LSA) und Wordcloud

## Voraussetzungen

- Python 3.9 oder höher
- Die in `requirements.txt` aufgeführten Bibliotheken

## Installation und Ausführung

```bash
# 1. Repository klonen
git clone https://github.com/MichiCorleone/NLP-Project.git
cd NLP-Project

# 2. (empfohlen) Virtuelle Umgebung anlegen
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Skript ausführen
python Projektcode.py
```

Beim ersten Start lädt das Skript automatisch die NLTK-Stoppwörter herunter.

## Projektstruktur

```
NLP-Project/
├── Projektcode.py          # Hauptskript mit der gesamten Pipeline
├── whatsapp_reviews.csv    # Verwendeter Datensatz
├── requirements.txt        # Python-Abhängigkeiten
├── README.md               # diese Datei
└── Erklärung.txt           # Konzept- und Erarbeitungstexte
```

## Ergebnisse (Kurzfassung)

Aus 4.577 Reviews wurden 1.325 negative Bewertungen (≤ 2 Sterne) analysiert.
LDA und LSA liefern fünf konsistente Themencluster, die sich inhaltlich folgenden
Bereichen zuordnen lassen:

- Konto- und Sperrproblematik (account, banned, spam)
- Funktions- und Update-Kritik (update, feature, new, status)
- Technische Probleme (cant, working, problem, slow)
- Kommunikationsqualität (call, chat, message, contact)
- Meta- / KI-Themen (meta, ai)

Eine ausführliche Diskussion der Ergebnisse, Limitationen sowie Verbesserungsvorschläge
findet sich im Full Abstract der Finalisierungsphase.

## Hinweise zur Anpassung

- **Sprache der Stoppwörter:** in `Projektcode.py` an der Stelle
  `stopwords.words("english")` einfach auf `"german"` oder eine andere Sprache umstellen
- **Filter-Schwellwert:** der Wert `rating <= 2` lässt sich beliebig anpassen, um
  z. B. nur 1-Stern-Reviews zu betrachten
- **Anzahl Themen:** Parameter `n_components=5` bei LDA und LSA steuert die Anzahl
  extrahierter Themen

## Autor

MichiCorleone – IU Internationale Hochschule, Cyber Security
