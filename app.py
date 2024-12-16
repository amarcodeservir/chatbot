import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk
from flask_cors import CORS
from flask import Flask, request, jsonify

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Download necessary NLTK data
nltk.download('stopwords')

# Load and prepare the dataset
DATASET_PATH = "generated_chatbot_data.xlsx"

try:
    df = pd.read_excel(DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please ensure the file exists in the project directory.")

# Validate dataset structure
required_columns = {'Query', 'Intent', 'Response'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset is missing required columns: {required_columns - set(df.columns)}")

# Drop unnecessary columns if they exist
if 'Chips' in df.columns:
    df.drop(columns=['Chips'], axis=1, inplace=True)

# Print the number of examples per intent
intents_summary = df.groupby('Intent').size()
print("Number of examples per intent:")
print(intents_summary)

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Preprocess the 'Query' column
df['Query'] = df['Query'].apply(preprocess_text)

# Split data into features and labels
X = df['Query']
y = df['Intent']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the queries using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train and evaluate RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)
rf_y_pred = rf_model.predict(X_test_vec)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))

# Train and evaluate LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)
lr_y_pred = lr_model.predict(X_test_vec)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_y_pred))

# Define API route for chatbot responses
@app.route('/get_response', methods=['POST'])
def api_get_response():
    try:
        # Get the user query from the request
        user_query = request.json.get('query', '')
        if not user_query:
            return jsonify({'response': "Please provide a query."}), 400
        
        # Preprocess the query and transform using TfidfVectorizer
        user_query_preprocessed = preprocess_text(user_query)
        user_query_tfidf = vectorizer.transform([user_query_preprocessed])
        
        # Predict the intent using Logistic Regression (you can switch to rf_model if needed)
        predicted_intent = lr_model.predict(user_query_tfidf)[0]
        
        # Retrieve the corresponding response
        response_row = df[df['Intent'] == predicted_intent]
        if not response_row.empty:
            response = response_row['Response'].iloc[0]
        else:
            response = "I'm sorry, I didn't understand that. Could you rephrase?"
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
