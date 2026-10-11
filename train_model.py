import re
import joblib
import pandas as pd

from collections import defaultdict, Counter

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# TASK 3 - PRODUCT CATEGORY CLASSIFIER
# FINAL TRAINING SCRIPT
# ============================================================


DATA_FILE = "products.csv"
MODEL_FILE = "product_category_model.pkl"


# ------------------------------------------------------------
# 1. Ucitavanje podataka
# ------------------------------------------------------------

print("Ucitavanje podataka...")

df = pd.read_csv(DATA_FILE)


# ------------------------------------------------------------
# 2. Ciscenje naziva kolona
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

df = df.rename(columns={
    "product ID": "product_id",
    "Product Title": "product_title",
    "Merchant ID": "merchant_id",
    "Category Label": "category",
    "_Product Code": "product_code",
    "Number_of_Views": "number_of_views",
    "Merchant Rating": "merchant_rating",
    "Listing Date": "listing_date"
})


# ------------------------------------------------------------
# 3. Uklanjanje praznih vrednosti
# ------------------------------------------------------------

df_clean = df.dropna(
    subset=["product_title", "category"]
).copy()

df_clean["product_title"] = (
    df_clean["product_title"]
    .astype(str)
    .str.strip()
)

df_clean["category"] = (
    df_clean["category"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# 4. Standardizacija kategorija
# ------------------------------------------------------------

category_mapping = {
    "fridge": "Fridges",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones"
}

df_clean["category"] = (
    df_clean["category"]
    .replace(category_mapping)
)

# Uklanjamo eventualne prazne naslove
df_clean = df_clean[
    df_clean["product_title"] != ""
].copy()


# ------------------------------------------------------------
# 5. Uklanjanje konfliktnih naslova
# ------------------------------------------------------------

# Proveravamo da li isti naslov pripada razlicitim kategorijama.
# Takve primere uklanjamo da model ne bi ucio kontradiktorne oznake.

title_category_counts = (
    df_clean
    .groupby("product_title")["category"]
    .nunique()
)

conflicting_titles = (
    title_category_counts[
        title_category_counts > 1
    ]
    .index
)

df_model = df_clean[
    ~df_clean["product_title"].isin(
        conflicting_titles
    )
].copy()


# ------------------------------------------------------------
# 6. Uklanjanje duplikata
# ------------------------------------------------------------

# Ako se isti naslov vise puta pojavljuje u istoj kategoriji,
# zadrzavamo samo jedan zapis.

df_model = df_model.drop_duplicates(
    subset=["product_title"],
    keep="first"
).copy()


# ------------------------------------------------------------
# 7. Feature engineering
# ------------------------------------------------------------

# Broj karaktera u naslovu
df_model["title_length"] = (
    df_model["product_title"]
    .str.len()
)

# Broj reci u naslovu
df_model["word_count"] = (
    df_model["product_title"]
    .str.split()
    .str.len()
)

# Broj cifara u naslovu
df_model["digit_count"] = (
    df_model["product_title"]
    .str.count(r"\d")
)

# Broj specijalnih znakova
df_model["special_char_count"] = (
    df_model["product_title"]
    .str.count(r"[^A-Za-z0-9\s]")
)

# Duzina najduze reci
df_model["longest_word_length"] = (
    df_model["product_title"]
    .apply(
        lambda text: max(
            len(word)
            for word in text.split()
        )
    )
)

numeric_features = [
    "title_length",
    "word_count",
    "digit_count",
    "special_char_count",
    "longest_word_length"
]


# ------------------------------------------------------------
# 8. Word-level TF-IDF
# ------------------------------------------------------------

print("Kreiranje word TF-IDF karakteristika...")

word_tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000
)

X_word = word_tfidf.fit_transform(
    df_model["product_title"]
)


# ------------------------------------------------------------
# 9. Character-level TF-IDF
# ------------------------------------------------------------

# Character n-grams pomazu modelu da prepozna oznake modela
# poput kgv39vl31g, wap28390gb, sbs8004po i slicno.

print("Kreiranje character TF-IDF karakteristika...")

char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000
)

X_char = char_tfidf.fit_transform(
    df_model["product_title"]
)


# ------------------------------------------------------------
# 10. Numericke karakteristike
# ------------------------------------------------------------

scaler = StandardScaler()

X_numeric = scaler.fit_transform(
    df_model[numeric_features]
)


# ------------------------------------------------------------
# 11. Spajanje svih karakteristika
# ------------------------------------------------------------

X_final = hstack([
    X_word,
    X_char,
    X_numeric
]).tocsr()

y = df_model["category"]


# ------------------------------------------------------------
# 12. Treniranje finalnog ML modela
# ------------------------------------------------------------

print("Treniranje finalnog Linear SVC modela...")

model = LinearSVC(
    max_iter=10000
)

model.fit(
    X_final,
    y
)


# ------------------------------------------------------------
# 13. Funkcija za izdvajanje model-kodova
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
# 14. Kreiranje model-code lookup pravila
# ------------------------------------------------------------

# Ideja:
# Ako se isti specificni model-kod pojavljuje vise puta
# i svaki put pripada istoj kategoriji, taj kod predstavlja
# veoma jak signal koji mozemo sacuvati uz ML model.

print("Kreiranje pouzdanih model-code pravila...")

code_categories = defaultdict(list)

for _, row in df_model.iterrows():

    codes = extract_model_codes(
        row["product_title"]
    )

    for code in codes:

        code_categories[code].append(
            row["category"]
        )


model_code_lookup = {}

for code, categories in code_categories.items():

    counts = Counter(categories)

    top_category, top_count = (
        counts.most_common(1)[0]
    )

    confidence = (
        top_count / len(categories)
    )

    # Pravilo cuvamo samo ako:
    # - kod postoji najmanje 2 puta
    # - svih 100% primera pripada istoj kategoriji
    if (
        len(categories) >= 2
        and confidence == 1.0
    ):
        model_code_lookup[
            code
        ] = top_category


print(
    "Broj pouzdanih model-code pravila:",
    len(model_code_lookup)
)


# ------------------------------------------------------------
# 15. Cuvanje kompletnog model paketa
# ------------------------------------------------------------

# U jednom .pkl fajlu cuvamo:
# - Linear SVC model
# - oba TF-IDF vectorizera
# - scaler
# - listu numerickih feature-a
# - kategorije
# - pouzdana model-code pravila

model_bundle = {
    "model": model,
    "word_tfidf": word_tfidf,
    "char_tfidf": char_tfidf,
    "scaler": scaler,
    "numeric_features": numeric_features,
    "categories": sorted(
        df_model["category"].unique()
    ),
    "model_code_lookup": model_code_lookup
}

joblib.dump(
    model_bundle,
    MODEL_FILE
)


# ------------------------------------------------------------
# 16. Zavrsna poruka
# ------------------------------------------------------------

print("\nModel je uspesno treniran.")
print(
    "Broj proizvoda:",
    len(df_model)
)
print(
    "Broj kategorija:",
    df_model["category"].nunique()
)
print(
    "Broj karakteristika:",
    X_final.shape[1]
)
print(
    "Broj model-code pravila:",
    len(model_code_lookup)
)
print(
    "Model je sacuvan kao:",
    MODEL_FILE
)