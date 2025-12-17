from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

classifier("I walked my dog around the neighborhood even i was tired")

classifier("I'm working since 4 hours")


classifier("I'm Thankful for all the blessings i have")


classifier("Education gives us the knowledge and skills we need to be successful in our careers and in our lives. It helps us to better understand the world around us and to make better decisions. Education is the foundation of our society and it is the key to a better future for all of us.")