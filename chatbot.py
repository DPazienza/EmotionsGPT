import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import whisper

# Check and download necessary resources
nltk.download('punkt', quiet=True)

# Step 1: Download and Load Dataset (GoEmotions and DailyDialog)
def download_datasets():
    datasets_dir = "datasets"
    os.makedirs(datasets_dir, exist_ok=True)

    try:
        print("\nLoading GoEmotions dataset...")
        go_emotions = load_dataset("go_emotions")
        print("GoEmotions dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading GoEmotions: {e}")
        raise

    try:
        print("\nLoading DailyDialog dataset...")
        daily_dialog = load_dataset("daily_dialog", trust_remote_code=True)
        print("DailyDialog dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading DailyDialog: {e}")
        raise

    return go_emotions, daily_dialog

# Step 2: Preprocessing Text
def preprocess_text(text):
    """Clean and preprocess text for emotion classification."""
    text = text.lower().strip()
    tokens = word_tokenize(text)
    return " ".join(tokens)

# Step 3: Load Pre-trained Model and Tokenizer (BERT fine-tuned for emotions)
def load_emotion_model():
    print("\nLoading pre-trained BERT model for emotion classification...")
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    print("Emotion model loaded successfully!")
    return emotion_pipeline

# Step 4: Load DialoGPT Model for Response Generation
def load_response_model():
    print("\nLoading DialoGPT model for response generation...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    print("DialoGPT model loaded successfully!")
    return model, tokenizer

# Step 5: Load Whisper Model for Speech-to-Text
def load_speech_model():
    print("\nLoading Whisper model for speech-to-text conversion...")
    model = whisper.load_model("base")
    print("Whisper model loaded successfully!")
    return model

# Step 6: Classify Emotions
def classify_emotion(emotion_pipeline, text):
    """Classify the emotion of the input text."""
    preprocessed_text = preprocess_text(text)
    results = emotion_pipeline(preprocessed_text)
    return results

# Step 7: Generate Chatbot Response and Maintain Context
def generate_response(response_model, tokenizer, user_input, context):
    """Generate a response based on the user's input and maintain context."""
    # Update context with the user's input
    context.append(f"User: {user_input}")
    context_text = " \n".join(context[-5:])

    # Encode the context
    input_ids = tokenizer.encode(context_text + tokenizer.eos_token, return_tensors="pt")

    # Generate a response
    output = response_model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True  # Enable sampling for diverse outputs
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the bot's response and update the context
    bot_response = response[len(context_text):].strip()
    context.append(f"Bot: {bot_response}")
    return bot_response

# Main Function to Run the Chatbot
def main():
    print("Initializing chatbot...\n")

    # Step 1: Download datasets
    go_emotions, daily_dialog = download_datasets()

    # Step 2: Load models
    emotion_pipeline = load_emotion_model()
    response_model, response_tokenizer = load_response_model()
    speech_model = load_speech_model()

    context = []  # Maintain conversation context

    print("\nChatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Take care.")
            break

        # Check if input is audio
        if user_input.startswith("audio:"):
            audio_path = user_input.split(":", 1)[1].strip()
            user_input = speech_model.transcribe(audio_path)["text"]

        # Classify emotion
        emotion_scores = classify_emotion(emotion_pipeline, user_input)
        print(f"Debug: Detected Emotion Scores - {emotion_scores}")

        # Generate response
        response = generate_response(response_model, response_tokenizer, user_input, context)

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()