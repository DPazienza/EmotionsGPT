from transformers import AutoTokenizer, AutoModelForCausalLM

class EmotionAwareChatModel:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.chat_history = ""

    def classify_emotion(self, emotion_scores):
        if isinstance(emotion_scores[0], list):
            emotion_scores = [item for sublist in emotion_scores for item in sublist]
        return sorted(emotion_scores, key=lambda x: x["score"], reverse=True)

    def generate_response(self, user_message, emotion_scores):
        emotions = self.classify_emotion(emotion_scores)
        dominant_emotion = emotions[0]['label']

        self.chat_history += f"User: {user_message}\n"
        if len(self.chat_history.split()) > 512:
            self.chat_history = ' '.join(self.chat_history.split()[-512:])

        prompt = f"""
        You are a friendly and empathetic virtual assistant. Your goal is to respond to the user's messages in a friendly, warm, and supportive tone. 

        Conversation so far:
        {self.chat_history}

        The user's emotion is: {dominant_emotion}.
        The user's input is: "{user_message}".

        Your response must:
        1. Directly address the user's message and emotion.
        2. Provide a friendly and encouraging follow-up question or comment.
        3. Avoid any unrelated or generic remarks. Do not mention yourself or provide irrelevant information.

        Respond as a supportive and friendly assistant:
        Assistant:
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=False
        )

        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=75,  # Risposte più concise
            do_sample=True,
            top_p=0.8,
            temperature=0.5,
            repetition_penalty=2.2,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.split("Assistant:")[-1].strip()

        
        self.chat_history += f"Assistant: {response}\n"
        return response
