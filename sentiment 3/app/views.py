from django.shortcuts import render, redirect
from django.urls import reverse
import pickle as pkl
from nltk.tokenize import word_tokenize


# Create your views here.
def index(request):
    return render(request,'index.html')

def load(fileName):
    file=open(fileName,'rb')
    data=pkl.load(file)
    file.close()
    return data


def predict(request):
    cv=load('cv.pkl')
    lm=load('lm.pkl')
    stopwordsList=load('stopwordsList.pkl')
    model=load('model.pkl')

    tweet=request.GET['tweets']

    tokenize_tweet=word_tokenize(tweet.lower())

    TweetList=[]
    for word in tokenize_tweet:
        if word.lower() not in stopwordsList:
            TweetList.append(word.lower())

    for i in range(len(TweetList)):
        TweetList[i]=lm.lemmatize(TweetList[i],pos='v')
        
    TweetList=' '.join(TweetList)
    TweetList=cv.transform([TweetList])
    prediction = model.predict(TweetList)

    
    if prediction[0]==0:
        msg='Irrelevant'
    elif prediction[0]==1:
        msg='Negative'
    elif prediction[0]==2:
        msg='Neutral'
    elif prediction[0]==3:
        msg='Positive'
    return render (request,'project.html', {'predict':msg})
    
     

