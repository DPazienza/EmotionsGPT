import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configurazione
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"
SELECTED_CLASSES = [8, 20, 24, 12, 15, 22]  # Emozioni principali come numeri (sadness, joy, love, anger, fear, surprise)
NEW_NUM_LABELS = len(SELECTED_CLASSES)

# Mappa le etichette alle classi 0-6
LABEL_MAPPING = {label: idx for idx, label in enumerate(SELECTED_CLASSES)}

# Caricamento del dataset da CSV
print("Caricamento del dataset GoEmotions dai file CSV...")
train_df = pd.read_csv("go_emotions_train.csv")
validation_df = pd.read_csv("go_emotions_validation.csv")
test_df = pd.read_csv("go_emotions_test.csv")

# Filtra e rimappa le etichette
print("Filtraggio e rimappatura delle etichette...")
def filter_and_remap(df):
    filtered_rows = []
    for _, row in df.iterrows():
        try:
            labels = [int(label) for label in row['labels'].strip('[]').split()]  # Splitta usando spazi
            valid_labels = [LABEL_MAPPING[label] for label in labels if label in LABEL_MAPPING]
            if valid_labels:
                row['labels'] = valid_labels[0]  # Prendi la prima etichetta valida
                filtered_rows.append(row)
        except Exception as e:
            print(f"Errore nel processare la riga: {row['labels']} - {e}")
    return pd.DataFrame(filtered_rows)

train_df = filter_and_remap(train_df)
validation_df = filter_and_remap(validation_df)
test_df = filter_and_remap(test_df)

# Converte i DataFrame in Dataset
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer
print("Caricamento del tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocessing e correzione delle etichette
def preprocess_data(examples):
    # Tokenizza con padding e truncation
    tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
    tokenized["labels"] = torch.tensor(examples["labels"], dtype=torch.long)  # Converti le etichette in tensori di tipo long
    return tokenized

print("Preprocessamento del dataset...")
train_dataset = train_dataset.map(preprocess_data, batched=True)
validation_dataset = validation_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

# Imposta il formato corretto per il dataset
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# DataCollator per batch dinamici
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Modello
print("Caricamento del modello pre-addestrato...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NEW_NUM_LABELS,
    ignore_mismatched_sizes=True
)

# Configurazione del Trainer
training_args = TrainingArguments(
    output_dir="./temp_results",  # Directory temporanea per evitare errori
    evaluation_strategy="epoch",  # Valutazione ad ogni epoca
    save_strategy="no",  # Non salvare i checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    load_best_model_at_end=False,
    report_to="none"  # Disabilita completamente i log
)

# Funzione di valutazione
print("Inizio addestramento...")
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

    # Verifica che ogni probabilità > 0.5 corrisponda a una label valida
    correct_predictions = 0
    total_predictions = 0
    for i, probs in enumerate(probabilities):
        predicted_labels = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        if set(predicted_labels).issubset(set([labels[i]])):
            correct_predictions += 1
        total_predictions += 1

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    prob_accuracy = correct_predictions / total_predictions

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "prob_accuracy": prob_accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Addestramento
trainer.train()

# Valutazione
print("Valutazione sul set di test...")
results = trainer.evaluate(eval_dataset=test_dataset)
print("Risultati:", results)

# Predizione con probabilità
print("Esempi di predizione con probabilità...")
def predict_with_probabilities(texts):
    encoded_inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**encoded_inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities

example_texts = [
    "I am so happy today!",
    "This is the worst day ever.",
    "I feel scared about the future.",
    "I am furious about what happened!"
]

probabilities = predict_with_probabilities(example_texts)
for i, text in enumerate(example_texts):
    print(f"Text: {text}")
    for j, prob in enumerate(probabilities[i]):
        print(f"  {SELECTED_CLASSES[j]}: {prob:.2f}")
    if torch.any(probabilities[i] > 0.5):
        print("  Predicted emotion(s) match labels with >50% probability")
    else:
        print("  No emotion predicted with >50% probability")

# Salvataggio del modello fine-tuned
print("Salvataggio del modello fine-tuned...")
trainer.save_model("./fine_tuned_go_emotions")
tokenizer.save_pretrained("./fine_tuned_go_emotions")
print("Fine-tuning completato!")
