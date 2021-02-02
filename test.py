import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
# import tensorflow as tf
import matplotlib.pyplot as plt

DATA_URL = ('./dataset/players_21.csv')

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data


@st.cache
def get_X(df):
    """Prepare X for data analysis purpose: X will contain cols that use for analysis only. e.g no name"""
    features=df.columns.tolist()
    features_dict={}
    counter=0
    for i in features:
      features_dict[i]=counter
      counter+=1

    # determine the clusters
    X = df.iloc[:,[features_dict['sofifa_id'],features_dict['age'],features_dict['height_cm'],features_dict['weight_kg'],features_dict['pace'],features_dict['shooting'],features_dict['passing'],features_dict['dribbling'],features_dict['defending'],features_dict['physic'],features_dict['attacking_crossing'],features_dict['attacking_finishing'],features_dict['attacking_heading_accuracy'],features_dict['attacking_short_passing'],features_dict['attacking_volleys'],features_dict['skill_dribbling'],features_dict['skill_curve'],features_dict['skill_fk_accuracy'],features_dict['skill_long_passing'],features_dict['skill_ball_control'],features_dict['movement_acceleration'],features_dict['movement_sprint_speed'],features_dict['movement_agility'],features_dict['movement_reactions'],features_dict['movement_balance'],features_dict['power_shot_power'],features_dict['power_jumping'],features_dict['power_stamina'],features_dict['power_strength'],features_dict['power_long_shots'],features_dict['mentality_aggression'],features_dict['mentality_interceptions'],features_dict['mentality_positioning'],features_dict['mentality_vision'],features_dict['mentality_penalties'],features_dict['mentality_composure'],features_dict['defending_standing_tackle'],features_dict['defending_sliding_tackle']]].values
    len(X)
    X=pd.DataFrame(X)
    #Drop rows with empty values
    X=X.dropna()
    X=X.iloc[:,:]
    return X

@st.cache
def get_elbow_curve(X):
    wcss = []
    for i in range(1, 14):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0,max_iter=300,n_init=10)# Note the initialisation used
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss


@st.cache
def compute_pca(data, n_components=2):
    """ PCA Calculationn: The PCA function for dimensionality reduction"""
    m, n = data.shape

    # Center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)

    #Find the eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to the sorted eigenvalues
    evals = evals[idx]
    # Select the top n eigenvectors and return their matrix product with the input data
    evecs = evecs[:, :n_components]
    return np.dot(evecs.T, data.T).T


dataset = load_data()
# Intro
st.write("""
Welcome to FIFA-21 Analyzer, this app allows you to analyze stats for FIFA-21 players.
""")

# TOP 10 feature
numerical_features = ['age', 'dob', 'height_cm', 'weight_kg', 'league_rank', 'overall', 'potential', 'value_eur', 'wage_eur', 'work_rate', 'release_clause_eur',  'joined', 'contract_valid_until', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'player_traits', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']
option = st.selectbox("Choose a stat", (numerical_features))
st.write(f'# Top 10 of {option}')
sort_descend = dataset.sort_values(option, ascending=False)
top_ten = sort_descend.head(10)
st.write(top_ten)

#### WSS#####
# WCSS is the sum of squares of the distances of each data point in all clusters to their respective centroids. The goal is to minimise the sum.
st.title('The Elbow Curve: Within-Cluster-Sum-of-Squares')
################

X = get_X(dataset)
wcss = get_elbow_curve(X)
st.line_chart(wcss)




# choose a nmber of clusters based on WSS
clusters_num = st.selectbox("Choose a number of clusters for K mean Analysis", (list(range(2,11))))

kmeans = KMeans(n_clusters = clusters_num, init = 'k-means++', random_state = 42,max_iter=300,n_init=10)
y_kmeans = kmeans.fit_predict(X)
### Search Players to alalyze on
array=np.array(dataset)
player_names = array[:,[0,2]]
# print(player_names[:,1])

player = st.selectbox("Search a player to make an analysis on", (player_names[:,1]))

# get cluster of a selected player
selected_player = np.where(player_names==player)
sofia_id = player_names[list(selected_player[0])][0,0]
index_of_player = np.where(X==sofia_id)[0]

y_kmeans_selected=y_kmeans[index_of_player]

#Storing players belonging to the same cluster as Lewandowski
predict_index=[]

for i in range(len(y_kmeans)):
  if(y_kmeans[i]==y_kmeans_selected):
    predict_index.append(i)

print(len(predict_index))

#Storing the fifa_ids of the players found above
X=X.reset_index()
print(X)

fifa_ids=[]
for i in predict_index:
  fifa_ids.append(X[0][i])

# print(fifa_ids[:5])

#Viewing the names of the players in the same cluster 
names=[]

for i in fifa_ids:
  names.append(dataset.loc[dataset['sofifa_id']==int(i)]['short_name'].values[0])


#Filtering out players who aren't in the same position
same_posiitons=[]
same_posiitons_ids=[]


players_positions = dataset.at[int(index_of_player),"player_positions"].split(', ')
print(players_positions)
for i in names:
  for position in players_positions:
    if (position in dataset['player_positions'][dataset.index[dataset['short_name']==i].values[0]]):
      same_posiitons.append(i)
      same_posiitons_ids.append(dataset.loc[dataset['short_name']==i]['sofifa_id'].values[0])



final_list=dataset.loc[dataset['short_name']==same_posiitons[0]][['short_name','club_name','nationality','overall','value_eur','age']]

st.write(final_list)
#Choosing the top 50 similar players based on overall
for i in range(1,30):
  temp=(dataset.loc[dataset['short_name']==same_posiitons[i]][['short_name','club_name','nationality','overall','value_eur','age']])
  final_list=final_list.append(temp)

  #Preparing data for visualisation
vis_dataframe=X.loc[X[0]==188545]

for i in range(1,50):
  vis_dataframe=vis_dataframe.append(X.loc[X[0]==same_posiitons_ids[i]])

vis_nparray=np.array(vis_dataframe)
vis_nparray=vis_nparray[:,1:]

st.write(final_list)



  
#Reducing the data dimensions for visualisation
X_reduced=compute_pca(vis_nparray,2)
print(X_reduced.shape)
players_to_visualise=same_posiitons[:14]
X_reduced_to_visualise=X_reduced[:14,:]

#Visualising the top 15 similar strikers graphically
st.write(f"# Top 15 Similar Plyaers to {player}")
fig = plt.figure()
plt.scatter(X_reduced_to_visualise[:, 0], X_reduced_to_visualise[:, 1])

for i, striker in enumerate(players_to_visualise):
    plt.annotate(striker, xy=(X_reduced_to_visualise[i, 0], X_reduced_to_visualise[i, 1]))
st.pyplot(fig)