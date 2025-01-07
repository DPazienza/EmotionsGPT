import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from lime.lime_text import LimeTextExplainer
import numpy as np

nltk.download('punkt')

class ExplainableEmotionsClassifier:
    def __init__(self):
        self.emotion_pipeline = self.load_emotion_model()
        self.explainer = LimeTextExplainer(class_names=self.get_emotion_labels())

    def preprocess_text(self, text):
        """Clean and preprocess text for emotion classification."""
        text = text.lower().strip()
        tokens = word_tokenize(text)
        return " ".join(tokens)

    def load_emotion_model(self):
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

    def explain_predictions(self, text, num_features=10):
        """Use LIME to explain emotion predictions for the input text."""
        def predict_proba(texts):
            """Helper function to predict probabilities for LIME."""
            results = [self.classify_emotion(text) for text in texts]
            probabilities = np.array([[score['score'] for score in result[0]] for result in results])
            return probabilities

        explanation = self.explainer.explain_instance(text, predict_proba, num_features=num_features)
        return explanation

if __name__ == "__main__":
    classifier = ExplainableEmotionsClassifier()

    sample_text = "I am feeling great and excited!"

    # Classify emotion
    emotion_results = classifier.classify_emotion(sample_text)
    print("Emotion Classification Results:", emotion_results)

    # Explain predictions using LIME
    lime_explanation = classifier.explain_predictions(sample_text)
    print("\nExplanation for Predictions:")
    for feature, weight in lime_explanation.as_list():
        print(f"Feature: {feature}, Weight: {weight}")
