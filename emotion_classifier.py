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

FINE_TUNED_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise", "neutral"]

class EmotionsClassifier:
    def __init__(self):
        self.emotion_pipeline = self.load_emotion_model()
        self.lime_explainer = LimeTextExplainer(class_names=FINE_TUNED_LABELS)
        self.shap_explainer = shap.Explainer(self.emotion_pipeline)
    
    def preprocess_text(self, text):
        """Clean and preprocess text for emotion classification."""
        text = text.lower().strip()
        tokens = word_tokenize(text)
        return " ".join(tokens)

    def load_emotion_model(self):
        #print("\nLoading fine-tuned model for emotion classification...")
        model_dir = "fine_tuned_go_emotions"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        print("Emotion model loaded successfully!")
        return emotion_pipeline

    def classify_emotion(self, text):
        """Classify the emotion of the input text."""
        preprocessed_text = self.preprocess_text(text)
        results = self.emotion_pipeline(preprocessed_text)
        return self.map_labels(results)
    
    def map_labels(self, results):
        """Map the model's output labels to the fine-tuned labels."""
        mapped_results = []
        for result in results:
            mapped_result = []
            for score in result:
                label_index = int(score['label'].split('_')[-1])  # Assuming labels are in the format 'LABEL_0', 'LABEL_1', etc.
                mapped_result.append({
                    'label': FINE_TUNED_LABELS[label_index],
                    'score': score['score']
                })
            mapped_results.append(mapped_result)
        return mapped_results
    
    def get_emotion_labels(self):
        """Retrieve emotion labels from the model's output."""
        return FINE_TUNED_LABELS
    
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
