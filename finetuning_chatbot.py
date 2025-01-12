from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import torch
import os

# Caricare il tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# Configurare il token di padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Usa il token di fine sequenza come token di padding

# Funzione per caricare e tokenizzare il dataset
def load_and_tokenize_dataset(csv_path):
    """Carica un dataset CSV e lo tokenizza."""
    data = pd.read_csv(csv_path)
    inputs = list(data['input'])
    outputs = list(data['output'])

    # Tokenizzare e allineare input/output
    tokenized_data = tokenizer(
        inputs,
        truncation=True,
        max_length=256,  # Mantieni una lunghezza massima per includere l'intero prompt
        padding='max_length',  # Il token di padding è configurato sopra
        return_tensors='pt'
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            outputs,
            truncation=True,
            max_length=256,  # Mantieni una lunghezza massima per evitare troncamenti
            padding='max_length',
            return_tensors='pt'
        )

    # Aggiungere i label al tokenized dataset
    tokenized_data["labels"] = labels["input_ids"]
    return tokenized_data

# Caricare i dataset preprocessati
train_dataset = load_and_tokenize_dataset("./processed_datasets/dailydialog_train_prompted.csv")
val_dataset = load_and_tokenize_dataset("./processed_datasets/dailydialog_validation_prompted.csv")

# Convertire i dati in Dataset di PyTorch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

train_dataset = CustomDataset(train_dataset)
val_dataset = CustomDataset(val_dataset)

# Verificare l'esistenza di un checkpoint valido
def get_checkpoint_dir(base_dir):
    required_files = ["config.json", "pytorch_model.bin"]
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if all(req_file in files for req_file in required_files):
                return root
    return None

checkpoint_dir = get_checkpoint_dir("./results")

# Caricare il modello dal checkpoint più recente o iniziare da zero
if checkpoint_dir:
    try:
        print(f"Caricamento del modello dal checkpoint: {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    except Exception as e:
        print(f"Errore nel caricamento del checkpoint: {e}")
        print("Ripristino del modello pre-addestrato.")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
else:
    print("Nessun checkpoint valido trovato. Caricamento del modello pre-addestrato.")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Configurare i parametri di training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,  # Salva il checkpoint ogni 500 passi
    save_steps=500,  # Salva il modello ogni 500 passi
    per_device_train_batch_size=32,  # Aumenta il batch size per sfruttare la GPU
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,  # Rimuovi l'accumulo dei gradienti per aggiornamenti più frequenti
    num_train_epochs=1,  # Riduci il numero di epoche per velocizzare il training
    learning_rate=5e-5,  # Leggermente più alto per velocizzare la convergenza
    warmup_steps=100,  # Riduci i warmup steps
    save_total_limit=1,  # Mantieni solo un checkpoint
    fp16=True,  # Usa floating point 16 se supportato
    gradient_checkpointing=True,  # Abilita il gradient checkpointing per risparmiare memoria
)

# Definire il Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Avviare il training
trainer.train(resume_from_checkpoint=checkpoint_dir is not None)

# Salvare il modello
model.save_pretrained("./finetuned_dialogpt")
tokenizer.save_pretrained("./finetuned_dialogpt")

# Testare il modello
test_sentence = "The user's emotion is happiness. The user's input is 'I'm feeling great today!'"
model.eval()
inputs = tokenizer(test_sentence, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
