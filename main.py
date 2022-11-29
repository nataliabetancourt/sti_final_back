import pandas as pd
import numpy as np
from numpy.linalg import norm
from flask import Flask, json, request
from flask_cors import CORS

# Declare the APP server instance
app = Flask(__name__)
# Enable CORS policies
CORS(app)

#Database
data = pd.read_csv('./data/final.csv')

#Database divided by sections to be used ahead
df_names = data.iloc[:, 0]
df_car = data.iloc[:, 1:6]
df_afinity = data.iloc[:, 1:7]

# Endpoint
@app.route("/", methods=["GET"])
def index():
  # args = request.args.to_dict()
  return json.dumps({
    "msg": "Hello Python REST API"
  })

@app.route("/get-group", methods=["GET"])
def get_group():
    args = request.args.to_dict();

#USER PART --------------
    #User inputs into list
    get_user = []

    get_user.append(float(args["r"]))
    get_user.append(float(args["n"]))
    get_user.append(float(args["d"]))
    get_user.append(float(args["c"]))
    get_user.append(float(args["m"]))

    new_user = pd.Series(get_user)

  #FIRST PART --------------
    #Get the nearest neighbors based on the caracteristics
    distance = []

    df_length = len(df_car.index)
    new_user_list = new_user.tolist()

    #Calculate the new users caracteristics with the database users
    for x in range(df_length):
          other_users = df_car.iloc[x].tolist()
          
          cosine = np.dot(new_user_list, other_users)/(norm(new_user_list)*norm(other_users))
          distance.append(cosine)
      
        
    #Create DataFrame with cosine similarity and names
    sim_caracteristics = pd.DataFrame()
    sim_caracteristics['Nombre'] = df_names.tolist()
    sim_caracteristics['Similitud'] = distance

    #Organize the list from highest to lowest
    order_sim = sim_caracteristics.sort_values(by=['Similitud'], ascending=False)
    nearest_neighbors_car = order_sim.head(3)


  #SECOND PART --------------
    #Get the nearest neighbors based on afinity

    #Get the index values of the nn of caracteristics
    indexNnCar = nearest_neighbors_car.index

    distance_afinity0 = []
    distance_afinity1 = []
    distance_afinity2 = []

    nn_car_users = []
    other_afi_users = []

    #Getting afinity info on caracteristic neighbors based on index
    for x in indexNnCar:
        nn_car_users.append(df_afinity.iloc[x])

    #Getting information of other users and adding to list
    for x in range(df_length):
        other_afi_users.append(df_afinity.iloc[x].tolist())
        
    #Similarity based on caracteristic neighbors and the rest, including afinity
    for i in range(len(nn_car_users)):
        for j in range(df_length):
            cosine = np.dot(nn_car_users[i], other_afi_users[j])/(norm(nn_car_users[i])*norm(other_afi_users[j]))
            
            #Add the similarity results based on the index of the first neighbors
            if i == 0:
                distance_afinity0.append(cosine)
            elif i == 1:
                distance_afinity1.append(cosine)
            else:
                distance_afinity2.append(cosine)

    #Create tables for each one
    sim_afi0 = pd.DataFrame()
    sim_afi1 = pd.DataFrame()
    sim_afi2 = pd.DataFrame()

    sim_afi0['Nombre'] = df_names.tolist()
    sim_afi0['Similitud'] = distance_afinity0

    sim_afi1['Nombre'] = df_names.tolist()
    sim_afi1['Similitud'] = distance_afinity1

    sim_afi2['Nombre'] = df_names.tolist()
    sim_afi2['Similitud'] = distance_afinity2
        
    #Organize the list from highest to lowest
    order_sim0 = sim_afi0.sort_values(by=['Similitud'], ascending=False)
    order_sim1 = sim_afi1.sort_values(by=['Similitud'], ascending=False)
    order_sim2 = sim_afi2.sort_values(by=['Similitud'], ascending=False)

    #Get the top names from each list
    nearest_neighbors_afi0 = order_sim0.head(5)
    nearest_neighbors_afi1 = order_sim1.head(5)
    nearest_neighbors_afi2 = order_sim2.head(5)

    #Add all the names to a final dataframe
    final = pd.concat([nearest_neighbors_afi0, nearest_neighbors_afi1, nearest_neighbors_afi2])
    final = final[~final.index.duplicated(keep='first')]
    final = final.sort_values(by=['Similitud'], ascending=False)

    #Create and add information to list that will be sent to front
    finalList = final.head(int(args["size"]))
    group = finalList["Nombre"].values.tolist()

    return json.dumps({
      "group": group})
  
# Execute the app instance wether is the main file
if __name__ == "__main__":
  app.run(debug=True)