import re
import joblib
import pandas as pd

from scipy.sparse import hstack


# ============================================================
# TASK 3 - PRODUCT CATEGORY CLASSIFIER
# INTERAKTIVNA PREDIKCIJA
# ============================================================


MODEL_FILE = "product_category_model.pkl"


# ------------------------------------------------------------
# 1. Ucitavanje sacuvanog model paketa
# ------------------------------------------------------------

print("Ucitavanje modela...")

bundle = joblib.load(MODEL_FILE)

model = bundle["model"]
word_tfidf = bundle["word_tfidf"]
char_tfidf = bundle["char_tfidf"]
scaler = bundle["scaler"]
numeric_features = bundle["numeric_features"]
categories = bundle["categories"]
model_code_lookup = bundle["model_code_lookup"]


# ------------------------------------------------------------
# 2. Funkcija za izdvajanje model-kodova
# ------------------------------------------------------------

def extract_model_codes(text):
    """
    Pronalazi duze alfanumericke oznake koje sadrze
    i slova i brojeve.

    Primeri:
    sbs8004po
    kgv39vl31g
    wap28390gb
    """

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    codes = []

    for token in tokens:

        has_letter = any(
            char.isalpha()
            for char in token
        )

        has_digit = any(
            char.isdigit()
            for char in token
        )

        if (
            len(token) >= 6
            and has_letter
            and has_digit
        ):
            codes.append(token)

    return codes


# ------------------------------------------------------------
# 3. Priprema karakteristika za ML model
# ------------------------------------------------------------

def prepare_features(title):

    temp = pd.DataFrame({
        "product_title": [title]
    })

    # Broj karaktera
    temp["title_length"] = (
        temp["product_title"]
        .str.len()
    )

    # Broj reci
    temp["word_count"] = (
        temp["product_title"]
        .str.split()
        .str.len()
    )

    # Broj cifara
    temp["digit_count"] = (
        temp["product_title"]
        .str.count(r"\d")
    )

    # Broj specijalnih znakova
    temp["special_char_count"] = (
        temp["product_title"]
        .str.count(r"[^A-Za-z0-9\s]")
    )

    # Duzina najduze reci
    temp["longest_word_length"] = (
        temp["product_title"]
        .apply(
            lambda text: max(
                len(word)
                for word in text.split()
            )
        )
    )

    # Word-level TF-IDF
    word_features = word_tfidf.transform(
        temp["product_title"]
    )

    # Character-level TF-IDF
    char_features = char_tfidf.transform(
        temp["product_title"]
    )

    # Numericke karakteristike
    numeric_features_scaled = scaler.transform(
        temp[numeric_features]
    )

    # Spajamo sve karakteristike
    combined_features = hstack([
        word_features,
        char_features,
        numeric_features_scaled
    ]).tocsr()

    return combined_features


# ------------------------------------------------------------
# 4. Predikcija kategorije
# ------------------------------------------------------------

def predict_category(title):
    """
    Hibridni pristup:

    1. Ako postoji veoma pouzdan poznati model-kod,
       koristi kategoriju sacuvanu za taj kod.

    2. Ako takav kod ne postoji, koristi Linear SVC model.
    """

    codes = extract_model_codes(title)

    # Prvo proveravamo poznate model-kodove
    for code in codes:

        if code in model_code_lookup:

            category = model_code_lookup[code]

            return category, "model-code lookup"

    # Ako kod nije poznat, koristi se ML model
    features = prepare_features(title)

    category = model.predict(
        features
    )[0]

    return category, "Linear SVC"


# ------------------------------------------------------------
# 5. Interaktivni rad
# ------------------------------------------------------------

print("\nModel je spreman.")
print("Unesi naziv proizvoda.")
print("Za izlaz upisi: exit\n")


while True:

    product_title = input(
        "Naziv proizvoda: "
    ).strip()

    # Izlazak iz programa
    if product_title.lower() == "exit":

        print("Program je zavrsen.")
        break

    # Ne dozvoljavamo prazan unos
    if not product_title:

        print(
            "Naziv proizvoda je prazan. "
            "Pokusaj ponovo.\n"
        )

        continue

    # Predikcija
    predicted_category, method = predict_category(
        product_title
    )

    print(
        "Predvidjena kategorija:",
        predicted_category
    )

    print(
        "Metod:",
        method,
        "\n"
    )