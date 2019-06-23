from flask import Flask, render_template, request
import movrec
import pandas as pd

app = Flask(__name__)

@app.route('/')
def form():
    return render_template("login.html")

@app.route("/welcome", methods=['POST'])
def welcome():
    result = request.form
    lis = movrec.movlis
    guser = result['username']
    gpass = result['password']
    word1 = movrec.dbuser[movrec.dbuser['user_name']==guser]['movie_title']
    word = word1.iloc[0]
    li = word.split('|')
    movt = []
    movp = []
    movtag = []
    for i in range(len(li)):
        print(li[i])
        movt.append(str(movrec.metadata1[movrec.metadata1['title'] == li[i]]['title'].values[0]))
        movp.append(str(movrec.metadata1[movrec.metadata1['title'] == li[i]]['poster_path'].values[0]))
        movtag.append(str(movrec.metadata1[movrec.metadata1['title'] == li[i]]['tagline'].values[0]))
    movtd = pd.DataFrame(list(zip(movt, movp, movtag)), columns=['title', 'poster_path', 'tagline'])
    #gpass = result['pass']
    for i in range(len(movrec.dbuser['user_name'])):
        if movrec.dbuser['user_name'][i] == result['username'] and movrec.dbuser['pass'][i] == result['password']:
            return render_template("welcome.html", a=result, lis = lis, movtd=movtd)
        elif i == len(movrec.dbuser['user_name']) - 1:
            return render_template("loginfail.html")

@app.route("/index", methods=['POST'])
def flas():
    result = request.form
    a = movrec.K_user(result.to_dict()['Name'])
    rec = result.to_dict()['Name']
    return render_template("index.html",a=a, recs = rec)

@app.route("/single")
def single():
    return render_template("singlepage.html")


if __name__ == '__main__':
    app.run()
