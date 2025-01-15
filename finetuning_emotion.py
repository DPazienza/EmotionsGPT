import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import pandas as pd
import os

# Configurazione
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"
FINE_TUNED_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise", "neutral"]
NUM_FINE_TUNED_LABELS = len(FINE_TUNED_LABELS)

# Caricamento del dataset
def load_and_filter_dataset(file_path, labels):
    print("Caricamento del dataset GoEmotions...")
    df = pd.read_csv(file_path)

    # Filtraggio delle etichette selezionate
    print("Filtraggio delle etichette selezionate...")
    df = df[["text"] + labels]

    # Rimozione dei record con tutte le emozioni pari a 0
    print("Rimozione dei record con emozioni tutte pari a 0...")
    df = df[df[labels].sum(axis=1) > 0]

    # Stampa del numero totale di record e conteggio per emozione
    print(f"Numero totale di record nel dataset filtrato: {len(df)}")
    print("Numero di record per emozione:")
    for label in labels:
        count = df[label].sum()
        print(f"  {label}: {int(count)}")

    # Conversione delle etichette in una lista binaria
    print("Preparazione delle etichette binarie...")
    df["labels"] = df.apply(lambda row: [row[label] for label in labels], axis=1)
    return Dataset.from_pandas(df)

# Dataset
train_validation_dataset = load_and_filter_dataset("go_emotions_dataset.csv", FINE_TUNED_LABELS)
dataset = train_validation_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
validation_dataset = dataset["test"]

# Tokenizer
print("Caricamento del tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenizzazione
def preprocess_data(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
    # Converte le etichette in tensori float con il formato corretto
    labels = torch.tensor(examples["labels"], dtype=torch.float)
    tokenized["labels"] = labels
    return tokenized

print("Preprocessamento del dataset...")
train_dataset = train_dataset.map(preprocess_data, batched=True)
validation_dataset = validation_dataset.map(preprocess_data, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# DataCollator per il padding dinamico
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Configurazione del Trainer
training_args = TrainingArguments(
    output_dir="./goemotions_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)

# Classe CustomTrainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(torch.float32)  # Converti le etichette in float32
        outputs = model(**inputs)
        logits = outputs.logits

        # Controlla che logits e labels abbiano la stessa forma
        assert logits.shape == labels.shape, f"Mismatch: logits {logits.shape}, labels {labels.shape}"

        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Verifica se esiste già il modello fine-tuned
if os.path.exists("./fine_tuned_go_emotions"):
    print("Modello fine-tuned trovato. Valutazione in corso...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./fine_tuned_go_emotions",
        num_labels=NUM_FINE_TUNED_LABELS,
        problem_type="multi_label_classification"
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    metrics = trainer.evaluate()

    # Calcolo di precision, recall e F1-score
    from sklearn.metrics import precision_recall_fscore_support

    # Ottieni i logit e le etichette previste
    predictions, labels, _ = trainer.predict(validation_dataset)
    predictions = torch.sigmoid(torch.tensor(predictions)) > 0.5  # Converti logit in predizioni binarie

    # Calcolo delle metriche
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    print("Risultati della valutazione:")
    print(metrics)
else:
    print("Inizio fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_FINE_TUNED_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

    # Salvataggio del modello fine-tuned
    print("Salvataggio del modello fine-tuned...")
    trainer.save_model("./fine_tuned_go_emotions")
    tokenizer.save_pretrained("./fine_tuned_go_emotions")
