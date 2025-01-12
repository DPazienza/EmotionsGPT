import pandas as pd

splits = {
    'train': 'simplified/train-00000-of-00001.parquet',
    'validation': 'simplified/validation-00000-of-00001.parquet',
    'test': 'simplified/test-00000-of-00001.parquet'
}

# Funzione per leggere e salvare i file Parquet in CSV
def save_parquet_to_csv():
    base_url = "hf://datasets/google-research-datasets/go_emotions/"

    for split_name, parquet_path in splits.items():
        print(f"Caricamento del dataset {split_name} da Parquet...")
        df = pd.read_parquet(base_url + parquet_path)

        csv_file = f"go_emotions_{split_name}.csv"
        print(f"Salvataggio del dataset {split_name} in {csv_file}...")
        df.to_csv(csv_file, index=False)

    print("Tutti i dataset sono stati salvati in formato CSV!")

# Esegui la funzione
if __name__ == "__main__":
    save_parquet_to_csv()
