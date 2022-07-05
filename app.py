import pickle
import re
import numpy as np
from flask import  Flask,request, jsonify, make_response
import tensorflow as tf
from flask_cors import CORS


with open('eurusdscaler.pkl','rb') as f:
    eurusdscaler = pickle.load(f)

with open('audusdscaler.pkl','rb') as f:
    audusdscaler = pickle.load(f)

with open('chfusdscaler.pkl','rb') as f:
    chfusdscaler = pickle.load(f)

with open('gbpusdscaler.pkl','rb') as f:
    gbpusdscaler = pickle.load(f)

with open('jpyusdscaler.pkl','rb') as f:
    jpyusdscaler = pickle.load(f)

with open('nzdusdscaler.pkl','rb') as f:
    nzdusdscaler = pickle.load(f)

eurusdmodel = tf.keras.models.load_model('eurusdmodel.h5')
audusdmodel = tf.keras.models.load_model('audusdmodel.h5')
chfusdmodel = tf.keras.models.load_model('chfusdmodel.h5')
gbpusdmodel = tf.keras.models.load_model('gbpusdmodel.h5')
jpyusdmodel = tf.keras.models.load_model('jpyusdmodel.h5')
nzdusdmodel = tf.keras.models.load_model('nzdusdmodel.h5')

def prediction(hours,inputs,model):
  pred = []
  inputs2 = np.reshape(inputs,(1,60,1))
  
  for i in range(hours):
    if i == 0:
      a = model.predict(inputs2)
      pred.append(a.tolist()[0][0])
      inputs2 = np.append(inputs2,pred[0])
      inputs2 = inputs2.tolist()
      inputs2 = np.reshape(inputs2,(1,61,1))
    elif i>0:
      b = model.predict(np.reshape(inputs2[0][i:],(1,60,1)))
      pred.append(b.tolist()[0][0])
      inputs2 = np.append(inputs2,pred[i])
      inputs2 = inputs2.tolist()
      inputs2 = np.reshape(inputs2,(1,(61+i),1))
  return pred


app = Flask(__name__)
CORS(app, supports_credentials=True)
@app.route("/",methods = ['GET'])
def hello():
    return jsonify({"response":"hello this is a forex app"})

@app.route("/eurusd",methods = ['POST'])
def eurusdpred():
  predprices =[]
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    hours = req['hours']
    inputs = eurusdscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(hours,inputs,eurusdmodel)
    arr = eurusdscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    for i in arr:
        predprices.append(i[0])
    res = make_response(jsonify({'predictions':predprices}))
    
    return res
  

@app.route("/audusd",methods = ['POST'])
def audusdpred():
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    hours = req['hours']
    inputs = audusdscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(hours,inputs,audusdmodel)
    arr = audusdscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    predprices = []
    for i in arr:
        predprices.append(i[0])
    res = make_response(jsonify({'predictions':predprices}))
    return res

@app.route("/chfusd",methods = ['POST'])
def chfusdpred():
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    hours = req['hours']
    inputs = chfusdscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(hours,inputs,chfusdmodel)
    arr = chfusdscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    predprices = []
    for i in arr:
        predprices.append(i[0])
    res =make_response(jsonify({'predictions':predprices}))
    return res

@app.route("/gbpusd",methods = ['POST'])
def gbpusdpred():
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    hours = req['hours']
    inputs = gbpusdscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(hours,inputs,gbpusdmodel)
    arr = gbpusdscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    predprices = []
    for i in arr:
        predprices.append(i[0])
    res =make_response(jsonify({'predictions':predprices}))
    return res

@app.route("/jpyusd",methods = ['POST'])
def jpyusdpred():
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    hours = req['hours']
    inputs = jpyusdscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(hours,inputs,jpyusdmodel)
    arr = jpyusdscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    predprices = []
    for i in arr:
        predprices.append(i[0])
    res =make_response(jsonify({'predictions':predprices}))
    return res

@app.route("/nzdusd",methods = ['POST'])
def nzdusdpred():
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    hours = req['hours']
    inputs = nzdusdscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(hours,inputs,nzdusdmodel)
    arr = nzdusdscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    predprices = []
    for i in arr:
        predprices.append(i[0])
    res =make_response(jsonify({'predictions':predprices}))
    return res

if __name__== "__main__":
    app.run(debug = True)