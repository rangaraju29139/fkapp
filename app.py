from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics;
import matplotlib.pyplot as plt

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def result():
    model_file = open('model.pkl', 'rb')
    fk_model = pickle.load(model_file)
    model_file.close()
    #if request.method == 'POST':
    text = request.form['article_text']

    data = [text]
    hash_vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
    check = hash_vectorizer.fit_transform(data)
    my_prediction =fk_model.predict(check)[0]
    return render_template('result.html',prediction=my_prediction)


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
	app.run(debug=True)
