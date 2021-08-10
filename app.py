from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from pprint import pprint



TMDB_API_KEY = 'YOUR_API_KEY'
RAPIDAPI_KEY ='YOUR_API_KEY'

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    username = db.Column(db.String(50), primary_key=True, unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)


db.create_all()

img_path = "http://image.tmdb.org/t/p/w500/{poster_path}"
cast_info_url = "https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}&language=en-US"

df2 = pd.read_csv('./model/tmdb.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

col_list = ['title', 'id']
df3 = pd.read_csv("./model/tmdb.csv", usecols=col_list)

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import string
import sys
import numpy as np
import datetime
tfds.load(name='imdb_reviews')
train_data, validation_data , test_data = tfds.load(name='imdb_reviews',split=['train[:60%]','train[60%:]','test'],as_supervised=True)
train_examples_data, train_labels_data = next(iter(train_data.batch(10)))
vector_model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" #Importing Pretrained model(Maps from text to 20-dimensional embedding vectors)on Tensorflow Hub
hub_layer = hub.KerasLayer(vector_model, input_shape=[],dtype=tf.string, trainable=True) #Creating a Keras layer of this model to use it in our CNN model.
ann_model = tf.keras.Sequential(); 
ann_model.add(hub_layer)#20 dim vector hence 20 neurons
ann_model.add(tf.keras.layers.Dense(16,activation = 'relu'))
ann_model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))
ann_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
ann_model.fit(train_data.shuffle(10000).batch(256),epochs=10,validation_data = validation_data.batch(256),verbose=1)
test_examples_data, test_labels_data = next(iter(test_data.batch(2000))) #Creating Batches of our Test Data
y_pred = ann_model.predict(test_examples_data) #Predicting the on the Test Data
y_pred = (y_pred > 0.5) #Converting y_pred into 0's and 1's depending on whether it is more than or lesser than 0.
y_test = np.reshape(test_labels_data,(-1,1)) #Reshaping the Test Label Data according to our y_pred

def get_recommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_indices]
    mov_id = df2['id'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title', 'Movie ID'])
    return_df['Title'] = tit
    return_df['Movie ID'] = mov_id
    return return_df


def get_single_movie_info(ids):
    api_key = TMDB_API_KEY
    img_path = "http://image.tmdb.org/t/p/w500"
    api_endpoint = "https://api.themoviedb.org/3/movie/"
    mov_dat_params = {
        "api_key": api_key,
        "language": "en-US"
    }
    temp = []
    response = requests.get(f"{api_endpoint}{ids}", params=mov_dat_params)
    data = response.json()
    temp.append(data['title'])
    temp.append(data['overview'])
    temp.append(data['vote_average'])
    tpp = []
    for gen in data['genres']:
        tpp.append(gen['name'])
    temp.append(tpp)
    temp.append(data['release_date'].split('-')[0])
    temp.append(data['runtime'])
    temp.append(f"{img_path}{data['poster_path']}")
    temp.append(data['imdb_id'])
    # pprint(temp)
    # print(temp[-1])
    url = "https://imdb8.p.rapidapi.com/title/get-user-reviews"
    querystring = {"tconst": f"{temp[-1]}"}
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': "imdb8.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()['reviews']
    revs = []
    analysis = []
    for i in range(10):
        tpp = str(data[i]['reviewText'])
        tpp.replace('"', "'")
        revs.append(tpp)
        review_word=ann_model.predict([tpp])[0][0] > 0.5
        if review_word:
            analysis.append("Positive")
        else:        
            analysis.append("Negative")
    # pprint(all_info)
    # print(len(all_info))
    # print(all_info[1])
    temp.append(revs)
    temp.append(analysis)
    print(analysis)
    return temp


def get_mult_movie_info(ids):
    api_key = TMDB_API_KEY
    img_path = "http://image.tmdb.org/t/p/w500"
    api_endpoint = "https://api.themoviedb.org/3/movie/"
    mov_dat_params = {
        "api_key": api_key,
        "language": "en-US"
    }
    all_movie_data = []
    for i in range(len(ids)):
        temp = []
        response = requests.get(f"{api_endpoint}{ids[i]}", params=mov_dat_params)
        data = response.json()
        # pprint(data)
        # print(f"Title: {data['title']}")
        temp.append(data['title'])
        # print(f"Description: {data['overview']}")
        temp.append(data['overview'])
        # print(f"Rating: {data['vote_average']} / 10")
        temp.append(data['vote_average'])
        # print("Genres: ", end="")
        tpp = []
        for gen in data['genres']:
            # print(gen['name'], end=" ")
            tpp.append(gen['name'])
        # print()
        temp.append(tpp)
        # print(f"Date: {data['release_date'].split('-')[0]}")
        temp.append(data['release_date'].split('-')[0])
        # print(f"Runtime: {data['runtime']} Minutes")
        temp.append(data['runtime'])
        # print(f"Image URL: {img_path}{data['poster_path']}")
        temp.append(f"{img_path}{data['poster_path']}")
        # print(temp)
        temp.append(data['imdb_id'])
        all_movie_data.append(temp)
        # print("\n\n")
    # print(all_movie_data)
    return all_movie_data


def get_cast_info(ids):
    api_key = TMDB_API_KEY
    img_path = "http://image.tmdb.org/t/p/w500/"
    cast_info_url = "https://api.themoviedb.org/3/movie/"
    cast_info_params = {
        "api_key": api_key,
        "language": "en-US"
    }
    temp = []
    response = requests.get(f"{cast_info_url}{ids}/credits?", params=cast_info_params)
    data = response.json()
    # pprint(data)
    if len(data['cast']) > 5:
        for i in range(5):
            tpp = []
            tpp.append(data['cast'][i]['name'])
            tpp.append(data['cast'][i]['character'])
            tpp.append(f"{img_path}{data['cast'][i]['profile_path']}")
            temp.append(tpp)
            # print(f"Actor: {data['cast'][i]['name']} Character: {data['cast'][i]['character']} Image Path: {img_path}{data['cast'][i]['profile_path']}")
    else:
        for i in range(len(data['cast'])):
            tpp = []
            tpp.append(data['cast'][i]['name'])
            tpp.append(data['cast'][i]['character'])
            tpp.append(f"{img_path}{data['cast'][i]['profile_path']}")
            temp.append(tpp)
            # print(f"Actor: {data['cast'][i]['name']} Character: {data['cast'][i]['character']}")
    # print(temp)
    # print("\n\n")
    return temp


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/login")
def login():
    what = request.args.get('do')
    print(what)
    if what == 'signup':
        head = "Get Started"
    else:
        head = "Log In"
    return render_template("loginpage.html", heading=head, what=what)


@app.route("/verfiy", methods=["GET", "POST"])
def verify():
    what = request.args.get('what')
    print(what)
    usrnm = request.form['username']
    pswd = request.form['password']
    if what == 'signup':
        new_user = User(
            username = usrnm,
            password = pswd
        )
        db.session.add(new_user)
        db.session.commit()
        data = ['Success!!', "Your account has been created successfully!!", 'Continue', "success"]
        return render_template("intermd.html", data=data)
    else:
        exists = db.session.query(User.username).filter_by(username=usrnm).first() is not None
        # print(exists)
        if exists:
            user_to_verify = User.query.get(usrnm)
            # print(user_to_verify.username)
            # print(user_to_verify.password)
            if (usrnm == user_to_verify.username) and (pswd == user_to_verify.password):
                data = ['Success!!', "You have been logged in successfully!!", 'Continue', "success"]
                return render_template("intermd.html", data=data)
            else:
                data = ['Oops!!', "Your Username and Password Do Not Match", 'Login Screen', "danger"]
                return render_template("intermd.html", data=data)
        else:
            data = ['Oops!!', "This Username Does Not Exist", 'Signup Screen', "danger"]
            return render_template("intermd.html", data=data)


@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        m_name = request.form['movie_name']
        m_name = m_name.title()
        #        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if m_name not in all_titles:
            return render_template('nfound.html', name=m_name)
        else:
            result_final = get_recommendations(m_name)
            names = []
            m_ids = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                m_ids.append(result_final.iloc[i][1])
            # print(m_ids)
            # m_ids = m_ids.tolist()
            for i in range(len(df3['title'])):
                if df3['title'].iloc[i] == m_name:
                    req_mov_id = df3['id'].iloc[i]
            # print(req_mov_id)
            req_mov_info = get_single_movie_info(req_mov_id)
            req_mov_cast_info = get_cast_info(req_mov_id)
            rec_mov_info = get_mult_movie_info(m_ids)
            # pprint(req_mov_info)
            # pprint(req_mov_cast_info)
            # pprint(rec_mov_info)
            return render_template('recommend.html', curr_mov_info=req_mov_info, curr_mov_cast=req_mov_cast_info, recm_mov_info=rec_mov_info)


if __name__ == '__main__':
    app.run(debug=True)





