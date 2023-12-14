import os
from flask import Flask, request, jsonify, render_template
import sys
import warnings
import nltk

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy import *
from urllib.parse import unquote

import numpy as np
import pandas as pd
import csv
import urllib.parse as parse
import pickle

app = Flask(__name__)
 # will hold the user input

# Loading Model ----------------------------------------------------------------------------------------------------

filename1 = 'lib/DecisionTreeClassifier.sav'
filename2 = 'lib/SVC.sav'
filename3 = 'lib/GaussianNB.sav'
filename4 = 'lib/KNeighborsClassifier.sav'
filename5 = 'lib/RandomForestClassifier.sav'
filename6 = 'lib/MLPClassifier.sav'

loaded_model1 = pickle.load(open(filename1, 'rb'))
loaded_model2 = pickle.load(open(filename2, 'rb'))
loaded_model3 = pickle.load(open(filename3, 'rb'))
loaded_model4 = pickle.load(open(filename4, 'rb'))
loaded_model5 = pickle.load(open(filename5, 'rb'))
loaded_model6 = pickle.load(open(filename6, 'rb'))


def get_feature_vectors(text):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(text)]
    max_epochs = 25
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    model.build_vocab(tagged_data)
    print("Model building started -/-")

    features = []

    for epoch in range(max_epochs):
        # print('Doc2Vec Iteration {0}'.format(epoch))
        print("*", sep=' ', end='', flush=True)
        model.random.seed(42)
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("lib/d2v.model")

    print("\nModel Saved")

    for i, line in enumerate(text):
        featureVec = [model.dv[i]]
        lineDecode = unquote(line)
        lineDecode = lineDecode.replace(" ", "")
        lowerStr = str(lineDecode).lower()

        # add feature for malicious HTML tag count
        feature1 = int(lowerStr.count('<link'))
        feature1 += int(lowerStr.count('<object'))
        feature1 += int(lowerStr.count('<form'))
        feature1 += int(lowerStr.count('<embed'))
        feature1 += int(lowerStr.count('<ilayer'))
        feature1 += int(lowerStr.count('<layer'))
        feature1 += int(lowerStr.count('<style'))
        feature1 += int(lowerStr.count('<applet'))
        feature1 += int(lowerStr.count('<meta'))
        feature1 += int(lowerStr.count('<img'))
        feature1 += int(lowerStr.count('<iframe'))
        feature1 += int(lowerStr.count('<input'))
        feature1 += int(lowerStr.count('<body'))
        feature1 += int(lowerStr.count('<video'))
        feature1 += int(lowerStr.count('<button'))
        feature1 += int(lowerStr.count('<math'))
        feature1 += int(lowerStr.count('<picture'))
        feature1 += int(lowerStr.count('<map'))
        feature1 += int(lowerStr.count('<svg'))
        feature1 += int(lowerStr.count('<div'))
        feature1 += int(lowerStr.count('<a'))
        feature1 += int(lowerStr.count('<details'))
        feature1 += int(lowerStr.count('<frameset'))
        feature1 += int(lowerStr.count('<table'))
        feature1 += int(lowerStr.count('<comment'))
        feature1 += int(lowerStr.count('<base'))
        feature1 += int(lowerStr.count('<image'))
        # add feature for malicious method/event count
        feature2 = int(lowerStr.count('exec'))
        feature2 += int(lowerStr.count('fromcharcode'))
        feature2 += int(lowerStr.count('eval'))
        feature2 += int(lowerStr.count('alert'))
        feature2 += int(lowerStr.count('getelementsbytagname'))
        feature2 += int(lowerStr.count('write'))
        feature2 += int(lowerStr.count('unescape'))
        feature2 += int(lowerStr.count('escape'))
        feature2 += int(lowerStr.count('prompt'))
        feature2 += int(lowerStr.count('onload'))
        feature2 += int(lowerStr.count('onclick'))
        feature2 += int(lowerStr.count('onerror'))
        feature2 += int(lowerStr.count('onpage'))
        feature2 += int(lowerStr.count('confirm'))
        feature2 += int(lowerStr.count('marquee'))
        # add feature for ".js" count
        feature3 = int(lowerStr.count('.js'))
        # add feature for "javascript" count
        feature4 = int(lowerStr.count('javascript'))
        # add feature for length of the string
        feature5 = int(len(lowerStr))
        # add feature for "<script"  count
        feature6 = int(lowerStr.count('<script'))
        feature6 += int(lowerStr.count('&lt;script'))
        feature6 += int(lowerStr.count('%3cscript'))
        feature6 += int(lowerStr.count('%3c%73%63%72%69%70%74'))
        # add feature for special character count
        feature7 = int(lowerStr.count('&'))
        feature7 += int(lowerStr.count('<'))
        feature7 += int(lowerStr.count('>'))
        feature7 += int(lowerStr.count('"'))
        feature7 += int(lowerStr.count('\''))
        feature7 += int(lowerStr.count('/'))
        feature7 += int(lowerStr.count('%'))
        feature7 += int(lowerStr.count('*'))
        feature7 += int(lowerStr.count(';'))
        feature7 += int(lowerStr.count('+'))
        feature7 += int(lowerStr.count('='))
        feature7 += int(lowerStr.count('%3C'))
        # add feature for http count
        feature8 = int(lowerStr.count('http'))

        # append the features
        featureVec = np.append(featureVec, feature1)
        # featureVec = np.append(featureVec,feature2)
        featureVec = np.append(featureVec, feature3)
        featureVec = np.append(featureVec, feature4)
        featureVec = np.append(featureVec, feature5)
        featureVec = np.append(featureVec, feature6)
        featureVec = np.append(featureVec, feature7)
        # featureVec = np.append(featureVec,feature8)
        # print(featureVec)
        features.append(featureVec)
    return features

# -------------


@app.route('/')
def home():
    return render_template('home.html')




@app.route('/about')
def about():
    return render_template('About.html')


@app.route('/services')
def services():
    return render_template('Services.html')


@app.route("/caption")
def caption():
    return render_template('detection.html')


@app.route("/editor")
def editor():
    return render_template('Enhance.html')


@app.route("/team")
def team():
    return render_template('Team.html')


@app.route("/descriptions")
def descriptions():
    return render_template('descriptions.html')


@app.route("/contact")
def contact():
    return render_template('Contact.html')


@app.route("/predict",methods=['GET','POST'])
def predict():
    URL = request.form['url']

    testXSS = []

    testXSS.append(URL)

    Xnew = get_feature_vectors(testXSS)
    # make a prediction
    # 1 DecisionTreeClassifier
    ynew1 = loaded_model1.predict(Xnew)
    # 2 SVC
    ynew2 = loaded_model2.predict(Xnew)
    # 3 GaussianNB
    ynew3 = loaded_model3.predict(Xnew)
    # 4 KNeighborsClassifier
    ynew4 = loaded_model4.predict(Xnew)
    # 5 RandomForestClassifier
    ynew5 = loaded_model5.predict(Xnew)
    # 6 MLPClassifier
    ynew6 = loaded_model6.predict(Xnew)

    xssCount = 0
    notXssCount = 0

    score = ((.175 * ynew1[0]) + (.15 * ynew2[0]) + (.05 * ynew3[0]) + (.075 * ynew4[0]) + (.25 * ynew5[0]) + (
                    .3 * ynew6[0]))
    print("Model Votes")
    print(ynew1[0])
    print(ynew2[0])
    print(ynew3[0])
    print(ynew4[0])
    print(ynew5[0])
    print(ynew6[0])
    print(score)

    if (ynew1 == 0):
        r1 = "URL is Non Malicious ✅"
    else:
        r1 = "URL is Malicious ❌"

    if (ynew2 == 0):
        r2 = "URL is Non Malicious ✅"
    else:
        r2 = "URL is Malicious ❌"

    if (ynew3 == 0):
        r3 = "URL is Non Malicious ✅"
    else:
        r3 = "URL is Malicious ❌"

    if (ynew4 == 0):
        r4 = "URL is Non Malicious ✅"
    else:
        r4 = "URL is Malicious ❌"

    if (ynew5 == 0):
        r5 = "URL is Non Malicious ✅"
    else:
        r5 = "URL is Malicious ❌"

    if (ynew6 == 0):
        r6 = "URL is Non Malicious ✅"
    else:
        r6 = "URL is Malicious ❌"

    if score >= .5:
        print("\033[1;31;1mXSS\033[0;0m => " + testXSS[0])
        xssCount += 1

        return render_template('detection.html', predictions="XSS Detected - URL IS NOT SAFE ❗❗️",
                               m1 = r1, m2 = r2, m3 = r3, m4 = r4, m5 = r5, m6 = r6, gurl = testXSS[0], scr = score)
    else:
        print("\033[1;32;1mNOT XSS\033[0;0m => " + testXSS[0])
        notXssCount += 1
        return render_template('detection.html', predictions="XSS NOT DETECTED - URL IS SAFE ☑️",
                               m1= r1, m2=r2, m3=r3, m4 = r4, m5=r5, m6=r6, gurl = testXSS[0], scr = score)


if __name__ == "__main__":
    app.run(debug=True, port=5555, host='0.0.0.0')
