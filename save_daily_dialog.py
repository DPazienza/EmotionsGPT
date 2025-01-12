import datasets
import pandas as pd
import os

# Mappatura delle emozioni
EMOTION_MAPPING = {
    0: "no emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

# Funzione per preprocessare il dataset DailyDialog e creare input con emozioni

def preprocess_and_save_dailydialog(output_dir):
    """Carica e preprocessa il dataset DailyDialog per creare un dataset con input, output e split."""
    # Caricare il dataset
    dataset = datasets.load_from_disk("./datasets/daily_dialog")

    # Prepara i dati per ogni split
    splits = {"train": [], "validation": [], "test": []}

    for split in ["train", "validation", "test"]:
        split_data = dataset[split]
        for example in split_data:
            dialog = example["dialog"]
            act = example["act"]
            emotion = example["emotion"]

            # Generazione degli input e output per ogni turno
            for i in range(len(dialog) - 1):
                user_message = dialog[i].strip()
                assistant_response = dialog[i + 1].strip()
                dominant_emotion = EMOTION_MAPPING.get(emotion[i], "no emotion")

                # Creazione del prompt
                prompt = f"""
                You are a friendly and empathetic virtual assistant. Your goal is to respond to the user's messages in a friendly, warm, and supportive tone. 

                Conversation so far:
                {user_message}

                The user's emotion is: {dominant_emotion}.
                The user's input is: "{user_message}".

                Your response must:
                1. Directly address the user's message and emotion.
                2. Provide a friendly and encouraging follow-up question or comment.
                3. Avoid any unrelated or generic remarks. Do not mention yourself or provide irrelevant information.

                Respond as a supportive and friendly assistant:
                Assistant:
                """

                # Aggiunta al dataset del rispettivo split
                splits[split].append({
                    "input": prompt.strip(),
                    "output": assistant_response
                })

    # Salvare ciascun split in un file CSV separato
    os.makedirs(output_dir, exist_ok=True)

    for split, data in splits.items():
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, f"dailydialog_{split}_prompted.csv")
        df.to_csv(output_path, index=False)
        print(f"{split.capitalize()} dataset salvato in: {output_path}")

    # Stampa di un esempio per verifica
    print("Esempio di record:")
    print(pd.DataFrame(splits["train"]).head())

# Esegui la funzione
output_directory = "./processed_datasets"
preprocess_and_save_dailydialog(output_directory)
