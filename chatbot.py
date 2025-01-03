from emotion_classifier import EmotionsClassifier
from chat_model import EmotionAwareChatModel

class EmotionsGPTChatbot:
    def __init__(self):
        self.context = []
        self.emotion_classifier = EmotionsClassifier()
        self.chat_model = EmotionAwareChatModel()
        self.full_transcript = [
            {"role": "system", "content": "You are a language model called Llama 3 created by Meta. Adapt your responses based on the user's emotions detected in the conversation. Keep your answers concise and under 300 characters. Do not use bold or asterisks as this will be passed to a text-to-speech service."},
        ]
        self.last_message = None

    
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

            # # Check if input is audio
            # if user_input.startswith("audio:"):
            #     audio_path = user_input.split(":", 1)[1].strip()
            #     user_input = self.speech_model.transcribe(audio_path)["text"]

            # Classify emotion
            emotion_scores = self.emotion_classifier.classify_emotion(user_input)
            print(f"Debug: Detected Emotion Scores - {emotion_scores}")

            # Generate response
            response =self.chat_model.generate_response(user_input, emotion_scores)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = EmotionsGPTChatbot()
    chatbot.run()
