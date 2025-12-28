# Task 1
# Convert Text to 5 Diff Languages Other Than Eng & Urdu 

import os
import warnings

# Suppress TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from transformers import pipeline

def main():
    inputText = "How is the weather today?"

    languageModels = {
        "German": "Helsinki-NLP/opus-mt-en-de",
        "French": "Helsinki-NLP/opus-mt-en-fr",
        "Spanish": "Helsinki-NLP/opus-mt-en-es",
        "Italian": "Helsinki-NLP/opus-mt-en-it",
        "Chinese": "Helsinki-NLP/opus-mt-en-zh"
    }

    print(f"Input Text: {inputText}\n" + "-"*30)

    for language, model_id in languageModels.items():
        try:
            print(f"Translating to {language}...")
            # We specify framework="pt" if you have torch, or "tf" for tensorflow
            # Using "pt" (PyTorch) is usually faster and has fewer warnings
            translator = pipeline("translation", model=model_id)
            
            result = translator(inputText)
            print(f"{language}: {result[0]['translation_text']}\n")
        except Exception as e:
            print(f"Could not translate {language}: {e}")

if __name__ == "__main__":
    main()