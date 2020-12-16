# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:01:00 2020

@author: mally
"""
import numpy as np
from tensorflow.keras import models
from flask import Flask,request

app=Flask(__name__)

model=models.load_model("Logistic_Prediction_Model1.h5")



@app.route('/',methods=["GET","POST"])
def predict():
    data=request.get_json(force=True)
    print(data)
    predi=model.predict(np.array([[data['a'],data['b'],data['c'],data['d'],data['e'],data['f'],data['g'],data['h'],data['i'],data['j']]]))
    output=predi
    print(output[0])    
    return str(output)

app.run()