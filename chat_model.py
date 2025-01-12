import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class EmotionAwareChatModel:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        try:
            # Configurazione del pipeline per il modello
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.bfloat16,  # Utilizza precisione bfloat16 per GPU
                device_map="auto"           # Mappa automaticamente alla GPU o CPU
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.pipe.model.resize_token_embeddings(len(self.tokenizer))

            self.chat_history = ""
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the model or tokenizer: {e}")

    def classify_emotion(self, emotion_scores):
        # Ordina le emozioni in base al punteggio
        if isinstance(emotion_scores[0], list):
            emotion_scores = [item for sublist in emotion_scores for item in sublist]
        return sorted(emotion_scores, key=lambda x: x["score"], reverse=True)

    def generate_response(self, user_message, emotion_scores):
        # Classifica le emozioni per determinare quella dominante
        emotions = self.classify_emotion(emotion_scores)
        dominant_emotion = emotions[0]['label']

        # Aggiungi il messaggio dell'utente alla cronologia
        self.chat_history += f"User: {user_message}\n"
        
        # Limita la cronologia per rimanere entro il limite di token
        if len(self.chat_history.split()) > 400:  # Mantieni solo le ultime 400 parole
            self.chat_history = ' '.join(self.chat_history.split()[-400:])

        # Crea il prompt con cronologia e contesto
        prompt = f"""
        The following is a conversation with a friendly and empathetic chatbot.

        Conversation history:
        {self.chat_history}

        User: {user_message}
        The user's emotion is: {dominant_emotion}

        Chatbot's response should:
        - Directly address the user's input.
        - Be friendly, empathetic, and supportive.
        - Avoid adding any external links or unrelated suggestions.
        - Provide a meaningful follow-up question or comment.

        Chatbot's response:
        """
        
        # Tokenizza e genera la risposta
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        outputs = self.pipe.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,  # Rendi il campionamento leggermente più restrittivo
            temperature=0.6,  # Riduci la creatività per risposte più coerenti
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decodifica e pulisci la risposta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.split("Chatbot's response:")[-1].strip()

        # Aggiorna la cronologia della chat
        self.chat_history += f"Assistant: {response}\n"

        return response
