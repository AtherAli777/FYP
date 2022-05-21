import nltk
import numpy as np
import json
import pickle

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Dropout
from tensorflow.keras.optimizers import SGD
import random

data_file = open('mental_health.json').read()
intents = json.loads(data_file)


words=[]
classes = []
sentences = []
ignore_words = ['?', '!','#','-','*','%','@']

lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize d
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        sentences.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize 
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
train = []
# create an empty array for our output
arr_output = [0] * len(classes)

# training set, bag of words for each sentence
for word in sentences:
    bow_list = []
    pattern_words = word[0]
    # lemmatize each word 
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words 
    for w in words:
        bow_list.append(1) if w in pattern_words else bow_list.append(0)
    
    output_row = list(arr_output)
    output_row[classes.index(word[1])] = 1
    
    train.append([bow_list, output_row])


random.shuffle(train)
train = np.array(train)

# create train and test 
X_train = list(train[:,0])
Y_train = list(train[:,1])

#Using DNN to create 3 nueron layers
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Y_train[0]), activation='softmax'))

# Compile model.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(X_train), np.array(Y_train), epochs=2000, batch_size=5, verbose=2)
model.save('model.h5', hist)
