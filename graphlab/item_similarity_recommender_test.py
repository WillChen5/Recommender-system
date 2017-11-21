import pandas as pd
import graphlab

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('../data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('../data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

#Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', 
	target='rating', similarity_type='cosine')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=50)
