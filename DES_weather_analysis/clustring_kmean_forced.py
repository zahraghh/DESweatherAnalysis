import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import sklearn.datasets, sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import sklearn_extra
from scipy import stats
from scipy.stats import kurtosis, skew
from collections import defaultdict
import statistics
from itertools import chain
from scipy.interpolate import interp1d
from collections import defaultdict
from nested_dict import nested_dict
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
JtokWh = 2.7778e-7
def kmedoid_clusters(path_test,scenario_genrated,name):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = editable_data['city']
    cluster_numbers= int(editable_data['Cluster numbers']) +2
    scenario_reduction_path= os.path.join(path_test,'ScenarioReduction')
    representative_days_path = scenario_reduction_path
    representative_day = {}
    representative_scenarios_list = []
    for represent in range(cluster_numbers):
        representative_day[represent] = pd.read_csv(os.path.join(representative_days_path,name+'Represent_days_modified_'+str(represent)+'.csv'))
        representative_scenario = representative_day[represent]['Electricity total (kWh)'].tolist() + representative_day[represent]['Heating (kWh)'].tolist() #+representative_day[represent]['GTI (Wh/m^2)'].tolist() + \
        #representative_day[represent]['Wind Speed (m/s)'].tolist() + representative_day[represent]['Electricity EF (kg/kWh)'].tolist()
        representative_scenarios_list.append(representative_scenario)
    #GTI_distribution = pd.read_csv(os.path.join(folder_path,'best_fit_GTI.csv'))
    #wind_speed_distribution = pd.read_csv(os.path.join(folder_path,'best_fit_wind_speed.csv'))
    scenario_probability = defaultdict(list)
    #laod the energy deamnd, solar, wind, and electricity emissions from scenario generation file
    scenario_probability = [1]*365
    features_scenarios = defaultdict(list)
    features_scenarios_list = []
    features_probability_list = []
    features_scenarios_nested = nested_dict()
    days= 365
    global k
    k = 0
    for i in range(days):
        data = scenario_genrated[i*24:(i+1)*24]
        data_1 = data['Total Electricity']
        data_2 = data['Total Heating']
        #Total electricity and heating
        daily_list =list(chain(data_1.astype('float', copy=False),data_2.astype('float', copy=False)))
        features_scenarios[k*days+i] = daily_list
        features_scenarios_nested[i] = features_scenarios[k*days+i]
        features_scenarios_list.append(features_scenarios[k*days+i])
        features_probability_list.append(scenario_probability[i])
        k = k+1
    A = np.asarray(features_scenarios_list)
    B = np.asarray(representative_scenarios_list)
    C = np.asarray(representative_scenarios_list+features_scenarios_list)

    #Convert the dictionary of features to Series
    standardization_data = StandardScaler()
    A_scaled = standardization_data.fit_transform(A)
    C_scaled = standardization_data.fit_transform(C)
    #print('Score of features', scores_pca)
    #print('Explained variance ratio',pca.explained_variance_ratio_)
    # Plot the explained variances
    # Save components to a DataFrame
    inertia_list = []
    search_optimum_cluster = editable_data['Search optimum clusters'] # if I want to search for the optimum number of clusters: 1 is yes, 0 is no
    kmeans = KMeans(n_clusters=cluster_numbers, n_init = 1, init = C_scaled[0:cluster_numbers]).fit(C_scaled)
    labels = kmeans.labels_
    clu_centres = kmeans.cluster_centers_
    z={i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    z_length = []
    all_representative_day = {}
    for i in range(kmeans.n_clusters):
        z_length.append(len(z[i])/len(labels))
        data_represent_days_modified={'Electricity total (kWh)': representative_day[i]['Electricity total (kWh)'],
        'Heating (kWh)': representative_day[i]['Heating (kWh)'],
        'Percent %': round(len(z[i])/len(labels)*100,4)}
        df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
        df_represent_days_modified.to_csv(os.path.join(representative_days_path,name+'Represent_days_modified_'+str(i)+ '.csv'), index=False)
        all_representative_day[i] = df_represent_days_modified
    return z_length,representative_day,all_representative_day
