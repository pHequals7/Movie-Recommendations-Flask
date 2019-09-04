from flask import Flask,render_template,request,Response,make_response
from scipy import spatial
import pandas as pd
import jellyfish as jf

app = Flask(__name__)

@app.route('/')
def home():
	print("hit")
	resp = make_response(render_template('home.html'))
	resp.headers['Access-Control-Allow-Origin'] = '*'
	resp.headers['X-Content-Type-Options'] = 'nosniff'
	return resp

@app.route('/predict',methods=['POST'])
def predict():

	def similarity(tensor,sent_embed2):
		return (1 - spatial.distance.cosine(sent_embed2,tensor))

	def typomatchjf(text,movie_name):
		return jf.jaro_distance(text,movie_name)


	def similar_movie(movie_name,topn=5):
		movie = pd.read_pickle('./df_movies3.pkl')
		try:
			se = movie.loc[movie_name.lower(),'embeddings'][:1][0]
			movie['similarity'] = movie['embeddings'].apply(lambda x:similarity(x,se))
			return movie.sort_values(by=['similarity'],ascending=False)[1:topn+1]
		except KeyError:
			movname = movie_name.lower()
			movie['typo'] = movie['Title'].apply(lambda x:typomatchjf(x.lower(),movname))
			x = movie.sort_values(by=['typo'],ascending=False)[:1]
			se2 = movie.loc[x.iloc[0]['Title'].lower(),'embeddings']
			movie['similarity'] = movie['embeddings'].apply(lambda x:similarity(x,se2))
			return movie.sort_values(by=['similarity'],ascending=False)[1:topn+1]	
	

	if request.method == 'POST':
		movie_name = request.form['movie']
		data = movie_name
		my_prediction = similar_movie(data)
		return render_template('predict.html',prediction = my_prediction,moviename = data)

@app.errorhandler(500)
def movie_not_found(e):
	return render_template('500.html'),500

if __name__ == '__main__':
	app.run()
