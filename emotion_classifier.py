import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import whisper
import assemblyai as aai
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import ollama

class EmotionsClassifier:
    def __init__(self):
        self.emotion_pipeline = self.load_emotion_model()

    def preprocess_text(self, text):
        """Clean and preprocess text for emotion classification."""
        text = text.lower().strip()
        tokens = word_tokenize(text)
        return " ".join(tokens)

    def load_emotion_model(self):
        #print("\nLoading pre-trained BERT model for emotion classification...")
        model_name = "bhadresh-savani/bert-base-uncased-emotion"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        print("Emotion model loaded successfully!")
        return emotion_pipeline

    def classify_emotion(self, text):
        """Classify the emotion of the input text."""
        preprocessed_text = self.preprocess_text(text)
        results = self.emotion_pipeline(preprocessed_text)
        return results
    


    # def download_datasets(self):
    #     datasets_dir = "datasets"
    #     os.makedirs(datasets_dir, exist_ok=True)

    #     try:
    #         print("\nLoading GoEmotions dataset...")
    #         go_emotions = load_dataset("go_emotions")
    #         print("GoEmotions dataset loaded successfully!")
    #     except Exception as e:
    #         print(f"Error loading GoEmotions: {e}")
    #         raise

    #     try:
    #         print("\nLoading DailyDialog dataset...")
    #         daily_dialog = load_dataset("daily_dialog", trust_remote_code=True)
    #         print("DailyDialog dataset loaded successfully!")
    #     except Exception as e:
    #         print(f"Error loading DailyDialog: {e}")
    #         raise

    #     return go_emotions, daily_dialog
