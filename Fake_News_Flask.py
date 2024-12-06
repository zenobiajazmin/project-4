from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Create Flask app
app = Flask(__name__)

# Initialize counters
fake_news_count = 0
real_news_count = 0

@app.route('/')
def home():
    # Render the homepage with initial percentages set to 0
    return render_template('index.html', fake_news_percent=0, real_news_percent=0)

@app.route('/predict', methods=['POST'])
def predict():
    global fake_news_count, real_news_count  # Use global counters

    # Define a custom threshold (e.g., 0.6 for "Real News")
    THRESHOLD = 0.6

    # Get user input from the form
    user_input = request.form['news_text']

    # Preprocess the input and predict probabilities
    input_vectorized = vectorizer.transform([user_input])
    probabilities = model.predict_proba(input_vectorized)[0]  # Get probabilities for each class

    # Use the threshold to determine the prediction
    if probabilities[1] >= THRESHOLD:  # Assuming class 1 is "Real News"
        real_news_count += 1
        result = "Real News"
    else:
        fake_news_count += 1
        result = "Fake News"

    # Calculate percentages
    total_predictions = fake_news_count + real_news_count
    fake_news_percent = (fake_news_count / total_predictions) * 100 if total_predictions > 0 else 0
    real_news_percent = (real_news_count / total_predictions) * 100 if total_predictions > 0 else 0

    # Render the results on the template
    return render_template(
        'index.html',
        prediction=result,
        user_input=user_input,
        fake_news_percent=round(fake_news_percent, 2),
        real_news_percent=round(real_news_percent, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)

