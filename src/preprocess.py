import pandas as pd
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 1. Loading the file
data = pd.read_csv('data/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# 2. Setup our cleaning tools
stop_words = stopwords.words('english')
ps = PorterStemmer()

# 3. The "Cleaning" Loop
cleaned_messages = []

for message in data['text']:

    '''Making lowercase'''
    message = message.lower()

    '''removing puntuation'''
    text_no_punct = ""

    # Looking at every single character in the message
    for char in message:
        # Check: Is this character a punctuation mark (like ! or ?)
        if char not in string.punctuation:
            text_no_punct += char

    message = text_no_punct
    
    '''Split into words, remove "the/is/a", and chop them down (stemming)'''
    # 1. Spliting the sentence into a list of individual words
    words = message.split()

    # 2. Creating a new empty list for our "cleaned" words
    useful_words = []

    # 3. Looking at every word one by one
    for w in words:
        if w not in stop_words:
            root_word = ps.stem(w)
            useful_words.append(root_word)

    # 4. Join the list back into a single sentence string
    message = " ".join(useful_words)
    
    # D. Put the words back into a sentence
    cleaned_messages.append(" ".join(useful_words))

# 4. Putting the cleaned text back into our data table
data['cleaned_text'] = cleaned_messages

# 5. Saving this to a new file 
data.to_csv('data/cleaned_spam.csv', index=False)

print("Finished! readable data generated for AI named cleaned_spam.csv")
print(data[['text', 'cleaned_text']].head())