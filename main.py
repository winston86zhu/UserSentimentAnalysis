# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os

import joblib
import nltk
from flask import request
import pickle
from flask import Flask, render_template
from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask

app = Flask(__name__)
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['input']
    selected_model = request.form['model']

    if selected_model == "" or selected_model == "analyzer":
        label = pre_built_model(text)
    elif selected_model == "bayes":
        label = bayes_model(text)
    elif selected_model == "svm":
        label = svm_model(text)
    else:
        label = "This model has not been built yet!"

    return render_template('index.html', variable=label)


# background process happening without any refreshing
@app.route('/background_process_task')
def background_process_task():
    dataset = pd.read_csv('./data/train.tsv', sep='\t', header=0)
    dataset['Phrase'] = dataset['Phrase'].str.strip().str.lower()

    from sklearn.model_selection import train_test_split
    x = dataset['Phrase']
    y = dataset['Sentiment']
    x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=5)
    token = RegexpTokenizer(r'[a-zA-Z]+')
    vec = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    x = vec.fit_transform(x)
    x_test = vec.transform(x_test)
    file = open('./temp/vec', 'wb')
    pickle.dump(vec, file)
    file.close()

    return "nothing"

# background process happening without any refreshing
@app.route('/background_process_task_svm')
def background_process_task_svm():
    dataset = pd.read_csv('./data/train.tsv', sep='\t', header=0)
    dataset['Phrase'] = dataset['Phrase'].str.strip().str.lower()
    dataset = dataset.sample(frac=.50)

    from sklearn.model_selection import train_test_split
    dataset['Phrase'] = dataset['Phrase'].str.strip().str.lower()
    dataset['Sentiment'] = dataset['Sentiment'].map(lambda a: convert_sentiment(a))
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    x = dataset['Phrase']
    y = dataset['Sentiment']
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.25, random_state=101)
    vectorizer.fit_transform(x)
    vectorizer.transform(x_test)

    file = open('./temp/vec_svm', 'wb')
    pickle.dump(vectorizer, file)
    file.close()

    return "nothing"

def bayes_model(input_text):
    bayes_model = load('./model/bayes_multinomial_model.pkl')
    with open(r"./temp/vec_bayes", "rb") as input_file:
        vec = pickle.load(input_file)

    result = bayes_model.predict(vec.transform([input_text]))
    print(result[0])
    return get_sentiment_category(result[0], 2)

def svm_model(input_text):
    svm_model = joblib.load('./model/svm_linear_model_2.pkl')
    with open(r"./temp/vec_svm", "rb") as input_file:
        vector = pickle.load(input_file)

    result = svm_model.predict(vector.transform([input_text]))
    print(result[0])
    return get_sentiment_category(result[0], 0)


def pre_built_model(input_text):
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = (sid.polarity_scores(str(input_text)))['compound']

    if score > 0:
        label = 'This input is positive :)'
    elif score == 0:
        label = 'This input is neutral -_-'
    else:
        label = 'This input is negative :('

    return label


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


def get_sentiment_category(score, threshold):
    if score > threshold:
        label = 'This input is positive :)'
    elif score == threshold:
        label = 'This input is neutral -_-'
    else:
        label = 'This input is negative :('

    return label

def convert_sentiment (score):
    score = int(score)
    if score > 2:
        label = 1
    elif score == 2:
        label = 0
    else:
        label = -1

    return label


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(port='8080', threaded=False, debug=False)
