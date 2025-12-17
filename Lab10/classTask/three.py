from transformers import pipeline
summarizer = pipeline("summarization")

ARTICLE = """
As companies increasingly adopt biometric time clock systems, you may wonder about the implications of using your fingerprint to clock in and out of work. While these systems promise enhanced security and efficiency, they raise important concerns regarding privacy and health.

Many employers view fingerprint scanners as a modern solution for tracking attendance, reducing instances of "time theft" where employees might punch in for one another. However, these machines rely on your unique biometric data, which brings risks associated with personal privacy and security.

With the recent COVID-19 pandemic, health concerns have also come to the forefront. Shared screens and fingerprint scanners can harbor viruses, emphasizing the need for regular cleaning and proper hygiene practices, such as handwashing or using sanitizer after use.

Critics highlight significant risks tied to biometric data, including potential identity theft and unauthorized surveillance. As privacy advocates like the ACLU point out, biometric information is permanent and cannot be changed like a password or credit card number. This immutability puts individuals at risk for long-term consequences if their data is compromised.

Various companies, from fast-food chains to airlines, have faced legal challenges regarding their use of biometric systems. Some argue that collective bargaining agreements with employee unions permit their use, while others have remained silent amid ongoing litigation.

As laws surrounding privacy and biometrics evolve, some states have enacted regulations to protect employees. In certain jurisdictions, you may have the right to refuse the use of biometric time clocks and even take legal action if necessary.
"""

summarizer(ARTICLE, max_length=130, min_length=30)


pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ur")


ARTICLE = """
 Reading books expands your knowledge.
"""

pipe(ARTICLE)


pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ur-en")


ARTICLE = """
"سلام۔",
"شکریہ۔",
"ہم جیت گئے۔",
"میں بیمار ہوں۔",
 "بہت خوب۔"

"""

pipe(ARTICLE)