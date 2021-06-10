# -*- coding: utf-8 -*-
"""Spam_Project_app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uZ6UugZukCxA1W4opoa3rlSEH5XxdkX6
"""

from flask import Flask
from flask import render_template, url_for, request
import pickle

app = Flask(__name__)

filename = 'spam_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', method=['POST'])
def predict():
  if request.method=='POST':
    msg = request.form['msg']
    data = [msg]
    vector = cv.transform(data).toarray()
    predicting = clf.predict(vector)
  return render_template('result.html',prediction=predicting)


if __name__ =='__main__':  
  app.run(debug = True)

dictionary = {1:'car', 2:'flower', 3:'plants', 4:'house'}
lst = [dictionary]
print(lst)

