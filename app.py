from flask import Flask,request, url_for, redirect, render_template
from article_recommendation import recommend_article

app = Flask(__name__)

@app.route('/')
def index():
    return render_template ('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_article():
    article = str(request.form.get('article'))

    #prediction
    final_result = recommend_article(article)    
    return render_template('index.html',final_result=final_result)

if __name__ == '__main__':
    app.run(debug=True)

    
