from transformers import LlamaTokenizer, LlamaForCausalLM
import torch  # noqa: F401
class EmotionAwareChatModel:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        # Carica il tokenizer e il modello
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        self.chat_history = ""  # Storia della chat

        # Aggiungi un token di padding se non esiste
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def classify_emotion(self, emotion_scores):
        """Trova le emozioni principali ordinate per punteggio."""
        # Ensure emotion scores are not nested inside an outer list
        if isinstance(emotion_scores[0], list):
            emotion_scores = [item for sublist in emotion_scores for item in sublist]
        
        sorted_emotions = sorted(emotion_scores, key=lambda x: x["score"], reverse=True)
        return sorted_emotions

    def generate_response(self, user_message, emotion_scores):
        """
        Genera una risposta basata sul messaggio dell'utente e sull'emozione rilevata.
        """
        # Classifica tutte le emozioni
        emotions = self.classify_emotion(emotion_scores)
        top_emotions = emotions[:3]
        top_emotions = [{**e, "score": round(e["score"], 2)} for e in top_emotions]
        dominant_emotion = top_emotions[0]["label"]
        # Aggiungi il messaggio dell'utente alla cronologia
        self.chat_history = ""  # Storia della chat
        self.chat_history += f"User: {user_message}\n"

        # Tokenizza e genera la risposta
        emotion_context = ", ".join([f"{e['label']}: {e['score']:.2f}" for e in emotions])
        prompt = (
            f"You are a friendly and empathetic conversational AI. Your goal is to engage in a natural and supportive conversation" 
            f"with the user, adapting your responses to their emotional state. \n"
            f"User's message::\n"
            f"{user_message}\n"
            f"The 3 dominant emotion is: {dominant_emotion}.\n\n"
            f"Conversation so far:\n{self.chat_history}\n"
            f"Respond appropriately, offering support and understanding as a friend would."
        )


        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length, padding='max_length')
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=3,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Aggiungi la risposta del bot alla cronologia
        self.chat_history += f"Bot: {response}\n"

        return response
