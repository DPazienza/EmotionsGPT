import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import lime
from lime.lime_text import LimeTextExplainer
import numpy as np

import shap

class EmotionsClassifier:
    def __init__(self):
        self.emotion_pipeline = self.load_emotion_model()
        self.lime_explainer = LimeTextExplainer(class_names=self.get_emotion_labels())
        self.shap_explainer = shap.Explainer(self.emotion_pipeline)
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
    
    def get_emotion_labels(self):
        """Retrieve emotion labels from the model's output."""
        sample_text = "I am happy"
        sample_result = self.classify_emotion(sample_text)
        return [emotion['label'] for emotion in sample_result[0]]
    
    def explain_predictions_lime(self, text, num_features=10):
        """Use LIME to explain emotion predictions for the input text."""
        def predict_proba(texts):
            """Helper function to predict probabilities for LIME."""
            results = [self.classify_emotion(text) for text in texts]
            probabilities = np.array([[score['score'] for score in result[0]] for result in results])
            return probabilities

        explanation = self.lime_explainer.explain_instance(text, predict_proba, num_features=num_features)
        return explanation
    

    def explain_predictions_shap(self, text):
        """Explain predictions using SHAP."""
        preprocessed_text = self.preprocess_text(text)
        shap_values = self.shap_explainer([preprocessed_text])
        return shap_values

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
