import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Loading the CLEANED data
data = pd.read_csv('data/cleaned_spam.csv')

# Drop any rows that became empty after cleaning
data = data.dropna()

# 2. Setup the "Math Converter" (CountVectorizer)
# This turns sentences into a grid of numbers
cv = CountVectorizer()
X = cv.fit_transform(data['cleaned_text'])
y = data['label'] 

# 3. 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create the "AI Brain" (Naive Bayes)
model = MultinomialNB()

# 5. TRAIN the model 
model.fit(X_train, y_train)

# 6. Check the Score
accuracy = model.score(X_test, y_test)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# 7. SAVE THE BRAIN AND THE CONVERTER
# We need both to predict new emails later
joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(cv, 'models/vectorizer.pkl')
print("Models and Vectorizer saved in models folder...")