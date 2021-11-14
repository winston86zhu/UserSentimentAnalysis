# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import nltk
from flask import request
from flask import jsonify
from flask import Flask, render_template

from flask import Flask

app = Flask(__name__)


@app.route('/')
def my_form():
	return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
	text = request.form['input']
	selected_model = request.form['model']

	if selected_model == "" or selected_model == "analyzer":
		label = pre_built_model(text)
	else:
		label = "This model has not been built yet!"

	return render_template('index.html', variable=label)


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	app.run(port='8088', threaded=False, debug=False)
