import fitz  # PyMuPDF
import spacy
import re

# Betöltjük az angol nyelvű SpaCy modellt
nlp = spacy.load("en_core_web_sm")

def extract_text_cleaned_nlp(pdf_path):
    doc = fitz.open(pdf_path)
    cleaned_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Split the text into individual lines for easier processing
        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            # Töröljük az üres sorokat
            if not line.strip():
                continue

            # Töröljük az oldalszámokat (pl. ha a sor csak számokat tartalmaz)
            if re.match(r'^\d+$', line.strip()):
                continue

            # SpaCy elemzést végzünk a sorokon
            spacy_doc = nlp(line)
            
            # Ha a sor csak számokat, rövid vagy általában címnek tűnik, kihagyjuk
            if len(line.strip()) < 3 or all(token.pos_ in ["PROPN", "NOUN"] for token in spacy_doc) and len(spacy_doc) < 5:
                continue

            # Ha a sor ismétlődő mintát tartalmaz (pl. szerző neve, könyvcím), eltávolítjuk
            if re.search(r'Heinlein|Robert A\. HEINLEIN|STRANGER IN A STRANGE LAND', line, re.IGNORECASE):
                continue

            # Ha átment az összes szűrőn, hozzáadjuk a tisztított szöveghez
            filtered_lines.append(line)

        cleaned_text += "\n".join(filtered_lines) + "\n"

    return cleaned_text

pdf_path = "/content/stranger_13_18.pdf"
cleaned_text = extract_text_cleaned_nlp(pdf_path)

# Az eredmény kiírása egy fájlba
with open("cleaned_text_nlp.txt", "w") as f:
    f.write(cleaned_text)
