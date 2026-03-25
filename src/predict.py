import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 1. Loading the "Saved Brain" and the "Translator"
model = joblib.load('models/spam_model.pkl')
cv = joblib.load('models/vectorizer.pkl')

# 2. Setup the same cleaning tools we used before
stop_words = stopwords.words('english')
ps = PorterStemmer()

def clean_my_text(text):
    # Lowercase
    text = text.lower()
    # Remove Punctuation
    clean_chars = ""
    for char in text:
        if char not in string.punctuation:
            clean_chars += char
    # Split, Remove Stopwords, and Stem
    words = clean_chars.split()
    useful_words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(useful_words)

# 3. The "Testing" Loop
print("--- Spam Detector is Online ---")
while True:
    user_input = input("\nEnter a message to check (or type 'quit'): ")
    
    if user_input.lower() == 'quit':
        break
        
    # Step A: Clean the user's input
    cleaned = clean_my_text(user_input)
    
    # Step B: Turn it into numbers (Vectorize)
    numeric_input = cv.transform([cleaned])
    
    # Step C: Ask the AI for a prediction
    prediction = model.predict(numeric_input)
    
    print(f"RESULT: This message is {prediction[0].upper()}")