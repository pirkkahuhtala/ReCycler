from http.server import BaseHTTPRequestHandler
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Esimerkkidata
data = {
    "text": [
        "Mihin laitan pahvit?",
        "Voiko muovipullot kierrättää?",
        "Mihin laitan paperit ja sanomalehdet?",
        "Entä biojätteet?",
        "Entä autonakku?",
        "Mihin vien vanhan akun?",
        "Missä voin kierrättää auton akkuja?",
        "Vanhat akut eivät kuulu biojätteeseen.",
        "Kuinka hävittää auton akku oikein?",
        "Mulla on pahvii tai tollaisia ruskeita pakkauksia, niin minne vien ne?",
        "Mulla on muovipullon, mihin se menee?",
        "Mihin laitan vanhat aikakausilehdet?",
        "Voiko paperipakkaukset laittaa kierrätykseen?",
        "Entä vanhat kirjat?",
        "Missä voin kierrättää muovipakkauksia?",
        "Mihin laitan vanhat sanomalehdet?",
        "Mihin voin laittaa vanhat mainokset?",
        "Missä voin kierrättää vanhat aikakausilehdet?",
    ],
    "label": [
        "Kartonkikeräys",  # Pahvi
        "Muovinkeräys",  # Muovi
        "Paperi",  # Paperi
        "Biojäte",  # Biojäte
        "sorikselle",  # Auton akku
        "sorikselle",  # Auton akku
        "sorikselle",  # Auton akku
        "sorikselle",  # Auton akku
        "sorikselle",  # Auton akku
        "Kartonkikeräys",  # Pahvi
        "Muovinkeräys",  # Muovi
        "Paperi",  # Paperi
        "Paperi",  # Paperi
        "Paperi",  # Paperi
        "Muovinkeräys",  # Muovi
        "Paperi",  # Paperi
        "Paperi",  # Uusi esimerkki
        "Paperi",  # Uusi esimerkki
    ],
}

# Muodosta DataFrame
df = pd.DataFrame(data)

# Poista rivit, joissa label on NaN
df = df.dropna(subset=["label"])

# Tulosta ladattu data
print("Ladattu data:")
print(df)

# Tarkista, onko DataFrame tyhjentynyt
if df.empty:
    print("Ei esimerkkejä datasetissä!")
    exit()

# Lataa tokenizer ja malli
model_name = "TurkuNLP/bert-base-finnish-cased-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Tokenisoi data
train_encodings = tokenizer(df["text"].tolist(), truncation=True, padding=True)


# Luo dataset-luokka
class RecyclingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Muutetaan labelit numeroiksi
label_map = {
    "Kartonkikeräys": 0,
    "Muovinkeräys": 1,
    "Biojäte": 2,
    "Paperi": 3,
    "sorikselle": 4,
}
df["label"] = df["label"].map(label_map)

# Tarkista, että kaikki labelit on mapattu oikein
df = df.dropna(subset=["label"])

# Muutetaan labelit kokonaisluvuiksi
df["label"] = df["label"].astype(int)

# Luo datasetti
train_dataset = RecyclingDataset(train_encodings, df["label"].tolist())

# Splittaa data koulutus- ja arviointidatasettiin
train_size = int(0.8 * len(df))
val_size = len(df) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# Aseta koulutusargumentit
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,  # Lisää epookkeja
    per_device_train_batch_size=4,
    logging_dir="./logs",
    learning_rate=2e-5,  # Muuta oppimisnopeutta
    evaluation_strategy="epoch",  # Arviointi jokaisen epookin jälkeen
    eval_steps=10,
)

# Luo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Kouluta malli
trainer.train()

# Tallenna malli ja tokenizer
model.save_pretrained("./kierratys_model")
tokenizer.save_pretrained("./kierratys_model")

# Luo pipeline luokittelua varten
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Testaa mallia esimerkkiteksteillä
texts_to_classify = [
    "Mihin laitan pahvit?",
    "Voiko muovipullot kierrättää?",
    "Entä biojätteet?",
    "Mulla on pahvii tai tollaisia ruskeita pakkauksia, niin minne vien ne?",
    "Mulla on tollanen autonakku, niin minne vien ne?",
    "Mulla on tässä vaihoja aikakausilehtiä, niin minne vien ne?",
    "Mihin laitan vanhat aikakausilehdet?",  # Uusi testi
    "Mihin laitan paperipakkaukset?",  # Uusi testi
]

# Muutetaan labelit takaisin selkokielisiksi
label_mapping_reverse = {
    0: "Kartonkikeräys",
    1: "Muovinkeräys",
    2: "Biojäte",
    3: "Paperi",
    4: "sorikselle",
}


def query(text):
    classification_results = classifier(text)
    predicted_label_index = int(classification_results[0]["label"].split("_")[-1])
    predicted_class = label_mapping_reverse[predicted_label_index]

    return {
        "text": text,
        "class": predicted_class,
        "score": classification_results[0]["score"],
    }
