import pandas as pd
import numpy as np
import flask
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import random
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# Load datasets
df_career = pd.read_csv("Model\Career QA Dataset.csv")
df_mock = pd.read_csv("Model\Mock-Interview-Questions.csv")
df_smalltalk = pd.read_csv("Model\customSmalltalkResponses_en.csv")

# Debugging: print column names
df_mock.rename(columns={'questionText': 'question', 'answerText': 'answer'}, inplace=True)
print("Mock Interview Questions columns:", df_mock.columns)
print("Small Talk Responses columns:", df_smalltalk.columns.tolist())

# Process small talk responses
df_smalltalk['answer'] = df_smalltalk.apply(
    lambda row: [ans for ans in [row['customAnswers__001'], row['customAnswers__002']] if pd.notna(ans) and ans != ''],
    axis=1
)
df_smalltalk = df_smalltalk[['action', 'answer']]
df_smalltalk.rename(columns={'action': 'question'}, inplace=True)

# Preprocess data
df_career['question'] = df_career['question'].str.lower().str.strip()
df_mock['question'] = df_mock['question'].str.lower().str.strip()
df_smalltalk['question'] = df_smalltalk['question'].str.lower().str.strip()

# Create small talk dictionary with all variations
smalltalk_dict = {}
for _, row in df_smalltalk.iterrows():
    question = row['question'].lower().strip()
    answers = row['answer']
    if isinstance(answers, list) and len(answers) > 0:
        smalltalk_dict[question] = random.choice(answers)

# Combine datasets for training
df_combined = pd.concat([df_career, df_mock, df_smalltalk], ignore_index=True)

# Train TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_combined['question'])
# Debugging: print shape of combined dataset
print("Combined dataset shape:", df_combined.shape)

# Debugging: print sample questions from combined dataset
print("Sample questions from combined dataset:", df_combined['question'].head())
# Save vectorizer and dataset
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(df_combined, open("dataset.pkl", "wb"))
pickle.dump(X, open("features.pkl", "wb"))
pickle.dump(smalltalk_dict, open("smalltalk_dict.pkl", "wb"))

# Initialize Flask app
#app = flask.Flask(__name__)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.json['question'].lower().strip()

    # Handle exit command
    if user_input == 'exit':
        return jsonify({"question": user_input, "answer": "Goodbye! Good luck with your interview preparation!"})

    # Check for exact small talk matches first
    smalltalk_dict = pickle.load(open("smalltalk_dict.pkl", "rb"))
    for pattern, response in smalltalk_dict.items():
        if user_input == pattern:
            return jsonify({"question": user_input, "answer": response})
        elif any(phrase in user_input for phrase in ['hi', 'hello', 'hey']):
            return jsonify({"question": user_input, "answer": response})
        elif any(phrase in user_input for phrase in ['thank you', 'thanks']):
            return jsonify({"question": user_input, "answer": "You're welcome!"})
        elif any(phrase in user_input for phrase in ['good night']):
            return jsonify({"question": user_input, "answer": "Good night! Sleep well and take care!"})
        elif any(phrase in user_input for phrase in ['good morning']):
            return jsonify({"question": user_input, "answer": "Good morning! Hope you have a great day ahead!"})
        elif any(phrase in user_input for phrase in ['good afternoon']):
            return jsonify({"question": user_input, "answer": "Good afternoon! How can I assist you today?"})
        elif any(phrase in user_input for phrase in ['good evening']):
            return jsonify({"question": user_input, "answer": "Good evening! I hope you had a wonderful day!"})
        elif any(phrase in user_input for phrase in ['how are you', 'how are you doing']):
            return jsonify({"question": user_input, "answer": "I'm just a bot, but I'm here to help you! How can I assist you today?"})

    # Fall back to career/mock interview answers
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    df = pickle.load(open("dataset.pkl", "rb"))
    X = pickle.load(open("features.pkl", "rb"))

    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match_idx = np.argmax(similarities)
    best_match_answer = df.iloc[best_match_idx]['answer']

    return jsonify({"question": user_input, "answer": best_match_answer})

if __name__ == '__main__':
    app.run(debug=True)
