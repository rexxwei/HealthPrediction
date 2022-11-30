
from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
import re


import flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# This page will have the sign up form
@app.route('/fastinput')
def fastinput():
    return render_template('fastinput.html')

@app.route('/pos')
def pos():
    return render_template('pos.html')

@app.route('/neg')
def neg():
    return render_template('neg.html')

@app.route('/predict', methods=['POST'])
def predict():
    # load the save model and scaler
    xgb_clf = joblib.load("xgb_clf.sav")
    scaler = joblib.load("scaler.save") 

    # transform the input text to numpy array
    form_input = request.form.to_dict()
    in_scores = []
    for k,v in form_input.items():
        in_scores.append(int(v))

    total = sum(in_scores)
    in_scores.append(total)
    X_test = [in_scores]
    X_scaled = scaler.transform(X_test)

    pred = xgb_clf.predict(X_scaled)

    if pred[0] == 0:
        return render_template('neg.html')
    else:
        return render_template('pos.html')

    return render_template('err.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
