Konzeptphase – Aufgabe 1: NLP – Techniken anwenden
Für dieses Portfolio wurde der Datensatz mit dem Namen „WhatsApp User Reviews“ (https://www.kaggle.com/datasets/sonalshinde123/whatsapp-user-reviews-dataset/data) verwendet.
Als erster Punkt wird zur Bereinigung des Textes alles in Kleinbuchstaben (lower()) umgewandelt und im Anschluss alle Stoppwörter entfernt und es wird eine Tokenisierung durchgeführt. 
Im Anschluss soll aus den unstrukturierten Textbausteinen numerische Vektoren gebildet werden. Dazu werden Techniken von Bag of Words und TF-IDF angewendet, um in erster Linie die Häufigkeit der auftretenden Wörter zu zählen und in weiterer Folge eine Wichtung durchzuführen.
Nachdem die häufigsten Wörter bzw. Begriffe ermittelt wurde, werden diese mittels Latent Dirichlet Allocation (LDA) und Latent Semantic Analysis (LSA) zu Themngruppen, jedes Thema wird durch charakteristische Wörter beschrieben, zusammengefasst.
Im Abschluss kann eine Visualisierung mittels Balkendiagrammen oder einer Work Cloud durchgeführt werden um so leichter Hauptthemen zu identifizieren.
Werkzeugübersicht (Python)
Zweck	Mögliche Bibliotheken
Textbearbeitung	nltk, scikit-learn
Themenmodellierung	scikit-learn
Visualisierung	matplotlib, wordcloud

