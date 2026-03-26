import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from art import *
print(text2art("---\t\t\tSPAM   DETECTOR---",font="small"))
print("\n***READY FOR PREDICTION***")

# 1. Loading the "Saved Brain" and the "Translator"
model = joblib.load('models/spam_model.pkl')
cv = joblib.load('models/vectorizer.pkl')

# 2. Setup the same cleaning tools we used before
stop_words = stopwords.words('english')
ps = PorterStemmer()

def clean_my_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove Punctuation
    text_no_punct = ""
    for char in text:
        if char not in string.punctuation:
            text_no_punct += char
    
    # 3. Split into words
    words = text_no_punct.split()

    # 4. Remove Stopwords and Stem
    useful_words = []
    for w in words:
        if w not in stop_words:
            root_word = ps.stem(w)
            useful_words.append(root_word)

    # 5. Join back together
    return " ".join(useful_words)

# 3. The "Testing" Loop
feedBack = input("Are you willing us to tell wether the AI model was right or not?Enter(y-yes/n-no) : ")
while True:
    label = ""
    user_input = input("\nEnter a message to check (or type 'quit'): ")
    
    if user_input.lower() == 'quit':
        break
        
    # Step A: Clean the user's input
    cleaned = clean_my_text(user_input)
    
    # Step B: Turn it into numbers (Vectorize)
    numeric_input = cv.transform([cleaned])
    
    # 1. Get the raw probabilities
    prob = model.predict_proba(numeric_input)[0]

    # 2. Get the final verdict (Spam/Ham)
    prediction = model.predict(numeric_input)[0]

    # Grab the scores
    ham_score = prob[0] * 100
    spam_score = prob[1] * 100

    # The Logic
    if spam_score > 50:
        print(f"Verdict: SPAM ({spam_score:.1f}%)")
        label = "spam"
    else:
        print(f"Verdict: HAM ({ham_score:.1f}%)")
        label = "ham"
    if feedBack == "y":
        correctness = input("Was the AI correct?Enter(y-yes,n-no) : ")
        if(correctness == "n"):
            if(label == "spam"):
                label = "ham"
            else:
                label = "spam"
        with open('data/spam.csv', 'a', encoding='latin-1') as f:
            f.write(f'\n{label},"{user_input}"')
def returnTOMain():
    if(feedBack == "y"):
        return True
    else:
        return False