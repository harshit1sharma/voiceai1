from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load FAQs from CSV file
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\ayush-python\main.py\ml\QnA5_modified.csv")
faqs = dict(zip(df['Question'], df['Answer']))

# Preprocess data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(faqs.keys())
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(faqs.keys())
max_len = max(len(seq) for seq in sequences)
X_train = pad_sequences(sequences, maxlen=max_len, padding='post')

# Define responses
responses = [faqs[q] for q in faqs.keys()]

# Convert responses to one-hot encoding
y_train = tf.keras.utils.to_categorical(range(len(responses)), num_classes=len(responses))

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=10, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=len(responses), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, verbose=0)

# Function to predict response
def predict_response(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    predicted_index = tf.argmax(model.predict(padded_sequence), axis=1).numpy()[0]
    return responses[predicted_index]

# Hardcoded credentials for demonstration purposes
VALID_USERNAME = 'Parul'
VALID_PASSWORD = 'pass123'

def authenticate(username, password):
    return username == VALID_USERNAME and password == VALID_PASSWORD

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def perform_login():
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')

    if authenticate(username, password):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = predict_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)


