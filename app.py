from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# ... (rest of your code)


app = Flask(__name__)
CORS(app)


def sentiment_analysis(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    scores = sid.polarity_scores(text)
    return scores

def sarcasm_detection(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    scores = sid.polarity_scores(text)
    if scores['compound'] > 0.5 and '!' in text:
        return True
    else:
        return False

def irony_detection(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    scores = sid.polarity_scores(text)
    if scores['compound'] < -0.5 and '!' in text:
        return True
    else:
        return False

def positive_or_negative(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    scores = sid.polarity_scores(text)
    if scores['compound'] > 0:
        return 'positive'
    elif scores['compound'] < 0:
        return 'negative'
    else:
        return 'neutral'

@app.route('/')
def index():
    return "Hello, this is the sentiment analyzer backend!"

@app.route('/perform_search', methods=['POST'])
def perform_search():
    user_input = request.form.get('user_input')

    sentiment_scores = sentiment_analysis(user_input)
    sarcasm = sarcasm_detection(user_input)
    irony = irony_detection(user_input)
    sentiment = positive_or_negative(user_input)

    return jsonify({'sentimentScores': sentiment_scores, 'sarcasm': sarcasm, 'irony': irony, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
