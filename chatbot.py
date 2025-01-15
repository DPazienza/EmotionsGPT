import os
import matplotlib.pyplot as plt
import seaborn as sns
from emotion_classifier import EmotionsClassifier
from chat_model import EmotionAwareChatModel
import numpy as np

class EmotionsGPTChatbot:
    def __init__(self):
        self.context = []
        self.emotion_classifier = EmotionsClassifier()
        self.chat_model = EmotionAwareChatModel()
        self.explanation_dir = 'explanation_graphs'
        os.makedirs(self.explanation_dir, exist_ok=True)

    def start_conversation(self):
        response = self.generate_response("Start the conversation with the user")
        print(f"Chatbot: {response}")

    def generate_response(self, user_input):
        return self.chat_model.generate_response(user_input)

    def plot_lime_explanation(self, lime_explanation, analyzed_text):
        features, weights = zip(*lime_explanation.as_list())
        plt.figure(figsize=(10, 6))
        plt.barh(features, weights, color="skyblue")
        plt.xlabel("Weight")
        plt.title(f"LIME Explanation\nAnalyzed Text: '{analyzed_text}'")
        plt.gca().invert_yaxis()

        filename = os.path.join(self.explanation_dir, f"lime_explanation_{analyzed_text[:15].replace(' ', '_')}.png")
        plt.savefig(filename)
        plt.close()
        print(f"LIME Explanation saved to {filename}")

    def plot_shap_explanation(self, features, shap_values, analyzed_text):
        """
        Plot a grouped bar chart with 7 bars for each feature, one for each emotion.
        :param features: List of feature names.
        :param shap_values: Array of SHAP values with shape (num_features, num_emotions).
        :param analyzed_text: The text being analyzed.
        """
        shap_values = np.array(shap_values)  # Ensure SHAP values are a NumPy array
        num_features, num_emotions = shap_values.shape

        # Define emotions and colors
        emotions = ["sadness", "joy", "love", "anger", "fear", "surprise", "neutral"]
        colors = sns.color_palette("tab10", num_emotions)

        x = np.arange(num_features)  # Feature indices
        bar_width = 0.15

        plt.figure(figsize=(14, 8))

        # Plot bars for each emotion
        for i in range(num_emotions):
            plt.bar(
                x + i * bar_width, 
                shap_values[:, i], 
                bar_width, 
                label=emotions[i], 
                color=colors[i]
            )

        # Add labels and title
        plt.xlabel("Features")
        plt.ylabel("SHAP Value")
        plt.title(f"SHAP Explanation\nAnalyzed Text: '{analyzed_text}'")
        plt.xticks(x + bar_width * (num_emotions / 2 - 0.5), features, rotation=45, ha="right")
        plt.legend(title="Emotions")

        # Save the plot
        filename = os.path.join(self.explanation_dir, f"shap_explanation_{analyzed_text[:15].replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"SHAP Explanation saved to {filename}")

    def run(self):
        print("Initializing chatbot...\n")
        print("\nChatbot is ready! Type 'exit' to quit.")

        while True:
            user_input = input("You: ")
            self.last_message = user_input

            if user_input.lower() == "exit":
                print("Chatbot: Goodbye! Take care.")
                break

            # Classify emotion
            emotion_scores = self.emotion_classifier.classify_emotion(user_input)
            print("Emotion Scores:")
            for emotion in emotion_scores[0]:
                print(f"{emotion['label']}: {emotion['score']:.2f}")

            # Explain predictions using LIME
            lime_explanation = self.emotion_classifier.explain_predictions_lime(user_input)
            print("\nLime Explanation for Predictions:")
            for feature, weight in lime_explanation.as_list():
                print(f"Feature: {feature}, Weight: {weight}")
            self.plot_lime_explanation(lime_explanation, user_input)

            # Explain predictions using SHAP
            shap_explanation = self.emotion_classifier.explain_predictions_shap(user_input)
            print("\nSHAP Explanation for Predictions:")
            shap_values = shap_explanation[0].values
            features = shap_explanation.data[0]

            # Check if SHAP values are multidimensional
            if len(shap_values.shape) > 1:
                self.plot_shap_explanation(features, shap_values, user_input)

            # Generate response
            response =self.chat_model.generate_response(user_input, emotion_scores)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = EmotionsGPTChatbot()
    chatbot.run()
