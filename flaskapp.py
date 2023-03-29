# -*- coding: utf-8 -*-
import pickle
import numpy as np
with open('LinearRegression.pkl','rb') as file:
    model =pickle.load(file)
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Regressiontemplate.html')

@app.route('/predict',methods=['POST'])
def predict():    
    input = [float(x) for x in request.form.values()]
    final_input = [np.array(input)]
    prediction = model.predict(final_input)
    return render_template('Regressiontemplate.html', prediction_text='Predicted Cost :{}'.format(prediction))

if __name__ == "__main__":
   app.run(debug=True)