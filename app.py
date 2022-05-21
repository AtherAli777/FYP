import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('mental_health.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):

    clean_sentence = nltk.word_tokenize(sentence)

    clean_sentence = [lemmatizer.lemmatize(word.lower()) for word in clean_sentence]
    return clean_sentence


def bow(sentence, words, show_details=True):
    # tokenize 
    sentence_words = clean_up_sentence(sentence)

    # bag of words 
    bow = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bow[i] = 1
                if show_details:
                    print ("In in distionary: %s" % w)
    return(np.array(bow))

def predict_class(sentence, model):

    p = bow(sentence, words,show_details=False)
    pred = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.20

    results = [[i,r] for i,r in enumerate(pred) if r>ERROR_THRESHOLD]
    # sort by strength of probability

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({"Intenets": classes[r[0]], "Probability": str(r[1])})
    return return_list

def randomResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = randomResponse(ints, intents)
    return res




                                            #<<<<<< Flask  >>>>>>>>
import pyrebase
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for,jsonify


app = Flask(__name__)       #Initialze flask constructor
app.static_folder = 'static'
app.template_folder = 'templates'
#Add your own details
config = {
      "apiKey": "AIzaSyDi6cFeemu7rzf0ZS3tMr47OzsmjEE0_9s",
      "authDomain": "chatauthentication-d125d.firebaseapp.com",

      "projectId": "chatauthentication-d125d",

      "storageBucket": "chatauthentication-d125d.appspot.com",

      "messagingSenderId": "295684616341",

      "appId": "1:295684616341:web:6e285eb33b4ed332a4ab9c",

      "measurementId": "G-T15NRCJKK4",

      "databaseURL": ""

}


@app.route("/get")
def get_bot_response():

    file1 = open('store.txt', 'a')

    userText = request.args.get('msg')
    
    response = chatbot_response(userText)
    
    file1.writelines(userText + '\n') 

    file1.close()         
            
    return response



#initialize firebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

#Initialze person as dictionary
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}


@app.route("/")
def home():
    return render_template("home.html")

#Login
@app.route("/login")
def login():
    return render_template("login.html")

#Sign up/ Register
@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/logout")
def logout():
    return render_template("login.html")

#chatbot page
@app.route("/chatbot")
def chatbot():
    if person["is_logged_in"] == True:
        return render_template("index.html", email = person["email"], name = person["name"])
    else:
        return redirect(url_for('login'))



@app.route("/result", methods = ["POST", "GET"])
def result():
	unsuccessful = 'Please check your credentials'
	successful = 'Login successful'
	if request.method == 'POST':
		email = request.form['email']
		password = request.form['pass']
        #name = request.form['name']

		try:
			user = auth.sign_in_with_email_and_password(email, password)
			return render_template('index.html')
		except:
			return render_template('login.html')

	return render_template('login.html')

#If someone clicks on register, they are redirected to /register
@app.route("/register", methods = ["POST", "GET"])
def register():
	unsuccessful = 'Please check your credentials'
	successful = 'Login successful'
	if request.method == 'POST':
		email = request.form['email']
		password = request.form['pass']
        #name = request.form['name']

		try:
			auth.create_user_with_email_and_password(email, password)
			return render_template('login.html')
		except:
			return render_template('signup.html')

	return render_template('login.html')


#Model Prediction
@app.route('/predict',methods=['POST','GET'])
def predict():
    import pandas as pd
    import joblib
    import re
    import matplotlib.pyplot as plt
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
    import pickle


    #('================Loading the Model================')
    model=joblib.load('ml_model.pkl')
    
    words = set(stopwords.words('english'))

    #test_data = str(request.args.get('msg'))
    test_data = open('store.txt', encoding='utf-8').read()
    test_data = test_data.lower()
    wl = WordNetLemmatizer()
    text=" ".join([wl.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", test_data).split() if i not in words]).lower()

    final_words=[]
    final_words_copy = []
    final_words.append(text)
    prediction = model.predict(final_words)
    prediction = prediction.tolist() # Numpy Array to List


    neu = 0
    neg = prediction.count(0)
    pos = prediction.count(1)

    return render_template('predict.html',neg=neg,pos=pos)

    '''
    print('Neg:', neg, ', Pos:', pos)
    if neg > pos:
        return render_template('predict.html',pred='Your sentiment found suicidal.\nProbability {}'.format(neg))
          
    elif pos > neg:
        return render_template('predict.html',pred='Your sentiment non found suicidal\nProbability {}'.format(pos))
            
    else:
        neu += 1
        print('Feedback: Neutral')
'''


if __name__ == "__main__":
    app.run()















'''
from flask import Flask, render_template, jsonify, request
import processor


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())



@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = processor.chatbot_response(the_question)

    return jsonify({"response": response })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
'''