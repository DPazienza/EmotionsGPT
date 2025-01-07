from emotion_classifier import EmotionsClassifier
from chat_model import EmotionAwareChatModel

class EmotionsGPTChatbot:
    def __init__(self):
        self.context = []
        self.emotion_classifier = EmotionsClassifier()
        self.chat_model = EmotionAwareChatModel()

    
    def start_conversation(self):
        response = self.generate_response("Start the conversation with the user")
        print(f"Chatbot: {response}")

    def generate_response(self, user_input):
        return self.chat_model.generate_response(user_input)
       


    def run(self):
        print("Initializing chatbot...\n")
        print("\nChatbot is ready! Type 'exit' to quit.")

        while True:
            # if self.last_message is None:
            #     self.start_conversation()
            

            user_input = input("You: ")
            self.last_message = user_input

            if user_input.lower() == "exit":
                print("Chatbot: Goodbye! Take care.")
                break


            # Classify emotion
            emotion_scores = self.emotion_classifier.classify_emotion(user_input)
            # print(emotion_scores)

            # # Explain predictions using LIME
            # lime_explanation = self.emotion_classifier.explain_predictions_lime(user_input)
            # print("\nLime Explanation for Predictions:")
            # for feature, weight in lime_explanation.as_list():
            #     print(f"Feature: {feature}, Weight: {weight}")

            # # Explain predictions using SHAP
            # shap_explanation = self.emotion_classifier.explain_predictions_shap(user_input)
            # print("\nSHAP Explanation for Predictions:")
            # shap_values = shap_explanation[0].values
            # shap_values = shap_explanation[0].values
            # features = shap_explanation.data[0]
            # for feature, value in zip(features, shap_values):
            #     print(f"Feature: {feature}, Weight: {value}")

            # Generate response
            response =self.chat_model.generate_response(user_input, emotion_scores)
            print(f"Chatbot: {response}")


if __name__ == "__main__":
    chatbot = EmotionsGPTChatbot()
    chatbot.run()
