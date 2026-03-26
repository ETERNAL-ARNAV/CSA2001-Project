# Technical Report: SMS Spam Classification System

**Author:** ARNAV SHARMA  
**Topic:** NLP-based Spam Detection using Naive Bayes  
**Date:**26 March 2026  

## 1. Objective
The primary goal of this project was to design and implement a functional Machine Learning pipeline capable of classifying SMS messages as "Spam" or "Ham" (legitimate). A secondary objective was to incorporate a "Human-in-the-Loop" feedback mechanism to allow the model to learn from its mistakes over time.

## 2. Technical Stack
- **Language:** Python 3.11+
- **Data Handling:** `pandas` for CSV manipulation.
- **NLP Processing:** `nltk` (Natural Language Toolkit) for stopword filtering and PorterStemming.
- **Machine Learning:** `scikit-learn` for CountVectorization and Multinomial Naive Bayes classification.
- **Model Persistence:** `joblib` for saving/loading serialized brain files (.pkl).
- **UI/UX:** `art` module for ASCII terminal branding and `Streamlit` for the web-based dashboard.

## 3. System Architecture
### Phase A: Preprocessing (`preprocess.py`)
Raw data from `spam.csv` is cleaned through a custom pipeline:
1. **Case Normalization:** Converts all text to lowercase to reduce vocabulary size.
2. **Punctuation Removal:** Iterates through every character to strip non-alphanumeric symbols.
3. **Tokenization:** Splits sentences into individual word units.
4. **Stemming:** Uses the Porter Stemmer to chop words to their root form (e.g., "running" becomes "run"), allowing the AI to group similar meanings.

### Phase B: Training (`train_model.py`)
1. **Vectorization:** Converts cleaned text into a numerical matrix where each column represents a unique word count.
2. **Data Splitting:** 80% of the data is used for training the "brain," while 20% is reserved for testing accuracy.
3. **Naive Bayes:** We chose the MultinomialNB algorithm because it is highly efficient for text-based probability calculations.

### Phase C: Inference & Feedback (`predict.py`)
The system loads the trained model and provides:
- **Real-time Prediction:** Classifies user input immediately.
- **Confidence Scoring:** Displays the probability percentage for both classes (Spam vs. Ham).
- **Active Learning:** If the user corrects the AI, the new data point is appended to the master CSV, allowing for a retrain cycle via `main.py`.

## 4. Performance Results
- **Initial Accuracy:** 97.76% on the test dataset.
- **Behavioral Note:** The model excels at identifying long, word-heavy spam but can be uncertain (near 50/50) on very short messages or messages containing only stop-words.

## 5. Future Enhancements
- Transition from `CountVectorizer` to `TfidfVectorizer` to better weight rare but significant spam keywords.
- Implement a threshold-based warning system for predictions with low confidence (e.g., < 60%).