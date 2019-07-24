from flask import Flask,render_template,request
import re
from scipy import spatial
from operator import itemgetter
import pandas as pd
import warnings
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings(action='ignore')

app = Flask(__name__)

@app.route('/')
def home():
	print("hit")
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	def clean_plot(text_list):
		clean_list = []
		for sent in text_list:
			sent = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-.:;<=>?@[\]^`{|}~"""), '',sent)
			sent = sent.replace('[]','')
			sent = re.sub('\d+',' ',sent)
			sent = sent.lower()
			clean_list.append(sent)
		return clean_list

	def img_ins(reco_list):
		img_reco_list = []
		for reco in reco_list:
			page = requests.get(reco[2])
			soup = BeautifulSoup(page.content, 'html.parser')
			img_link = soup.find('meta',{'property':"og:image"})
			if img_link is not None:
				imgurl = img_link.get('content')
				reco = reco+(imgurl,)
				img_reco_list.append(reco)
			else:
				imgur = 'https://cutt.ly/EF53JZ'
				reco = reco+(imgur,)
				img_reco_list.append(reco)
		return img_reco_list

	def similar_movie(movie_name,topn=5):
	    movie = pd.read_pickle('./movie_df.pkl')
	    movie['Title_srch'] = movie['Title'].apply(lambda x: x.lower())
	    movie.set_index('Title_srch', inplace=True)
	    sent_embed2 = movie.loc[movie_name.lower(),'embeddings']
	    similarities = []
	    #for tensor,title,wikiurl,plot in zip(movie['embeddings'],movie['Title'],movie['Wiki Page'],movie['Plot']):
	    for index,row in movie.iterrows():
		    cos_sim = 1 - spatial.distance.cosine(sent_embed2,row['embeddings'])
		    similarities.append((row['Title'],cos_sim,row['Wiki Page'],str(row['Plot'][:1500]+'....')))
	    reco_list = sorted(similarities,key=itemgetter(1),reverse=True)[1:topn+1]
	    return img_ins(reco_list)

	if request.method == 'POST':
		movie_name = request.form['movie']
		data = movie_name
		my_prediction = similar_movie(data)
		return render_template('predict.html',prediction = my_prediction,moviename = data)

if __name__ == '__main__':
	app.run()

