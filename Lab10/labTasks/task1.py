# Task 1
# Convert Text to 5 Diff Languages Other Than Eng & Urdu 

from transformers import pipeline

inputText = "How is the weather today?"

languageModels = {
    "German": "Helsinki-NLP/opus-mt-en-de",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "Italian": "Helsinki-NLP/opus-mt-en-it",
    "Chinese": "Helsinki-NLP/opus-mt-en-zh"
}

print(languageModels.items())

for language, model in languageModels.items():
    translator = pipeline("translation", model=model)
    translatedText = translator(inputText)[0]["translation_text"]
    print(f"{language}: {translatedText}")