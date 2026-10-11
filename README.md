# Product Category Classifier

Machine learning projekat za automatsku klasifikaciju proizvoda u odgovarajucu kategoriju na osnovu naziva proizvoda.

## Autor

Igor Jovanovic

## Cilj projekta

Cilj projekta je razvoj modela koji na osnovu teksta iz kolone `Product Title` predvidja odgovarajucu kategoriju proizvoda.

Dataset sadrzi proizvode iz 10 kategorija, ukljucujuci:

- CPUs
- Digital Cameras
- Dishwashers
- Freezers
- Fridge Freezers
- Fridges
- Microwaves
- Mobile Phones
- TVs
- Washing Machines

## Struktura projekta

```text
Task3_ProductClassifier/
│
├── products.csv
├── product_classifier_analysis.ipynb
├── train_model.py
├── predict_category.py
├── product_category_model.pkl
└── README.md