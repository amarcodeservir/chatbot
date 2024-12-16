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
from flask_cors import CORS  # Import CORS
from flask import Flask, request, jsonify
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download NLTK stopwords if not already installed
nltk.download('stopwords')

# Load the dataset
df = pd.read_excel(r"C:\Users\ritan\OneDrive\Desktop\New folder (2)\generated_chatbot_data.xlsx")
print(df)
# Drop the 'Chips' column
df.drop(columns=['Chips'], axis=1, inplace=True)

# Check the number of examples per intent
intents_summary = df.groupby('Intent').size()
print("Number of examples per intent:")
print(intents_summary)

# Preprocess the queries
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing to the 'Query' column
df['Query'] = df['Query'].apply(preprocess_text)

# Split data into features and labels
X = df['Query']
y = df['Intent']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the queries
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)

# Evaluate RandomForestClassifier
rf_y_pred = rf_model.predict(X_test_vec)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))

# Train LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

# Evaluate LogisticRegression
lr_y_pred = lr_model.predict(X_test_vec)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_y_pred))

# Map user queries to responses
def get_response(user_query, model='lr'):
    # Preprocess the user query
    user_query_preprocessed = preprocess_text(user_query)
    user_query_tfidf = vectorizer.transform([user_query_preprocessed])
    
    # Predict intent based on the selected model
    if model == 'rf':  # Use Random Forest
        predicted_intent = rf_model.predict(user_query_tfidf)[0]
    elif model == 'lr':  # Use Logistic Regression
        predicted_intent = lr_model.predict(user_query_tfidf)[0]
    else:
        return "Error: Invalid model selected!"
    
    # Retrieve the corresponding response
    response_row = df[df['Intent'] == predicted_intent]
    if not response_row.empty:
        return response_row['Response'].iloc[0]
    else:
        return "I'm sorry, I didn't understand that. Could you rephrase?"



# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input, model='lr')  # Use 'rf' for Random Forest
    print(f"Chatbot: {response}")


# API route for chat responses
@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.json['query']
    user_query = preprocess_text(user_query)
    query_vec = vectorizer.transform([user_query])
    predicted_intent = model.predict(query_vec)[0]
    response = response_dict.get(predicted_intent, "Sorry, I didn't understand that.")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)




