import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import datetime
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
import DES_weather_analysis
from DES_weather_analysis import clustring_kmean_forced, clustring_kmediod_PCA_operation, EPW_to_csv,solar_irradiance,solar_position
from DES_weather_analysis.solar_irradiance import aoi, get_total_irradiance
from DES_weather_analysis.solar_position import get_solarposition
JtokWh = 2.7778e-7
def kmedoid_clusters(path_test,mode):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    scenario_reduction_path= os.path.join(path_test,'ScenarioReduction')
    scenarios_path = os.path.join(path_test,'ScenarioGeneration')
    if not os.path.exists(scenario_reduction_path):
        os.makedirs(scenario_reduction_path)
    representative_days_path = scenario_reduction_path
    num_scenario = 0
    num_scenarios = int(editable_data['num_scenarios'])
    city=editable_data['city']
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    altitude = float(editable_data['Altitude']) #SLC altitude m
    surf_tilt = float(editable_data['solar_tilt']) #panels tilt degree
    surf_azimuth = float(editable_data['solar_azimuth']) #panels azimuth degree
    idf_names= []
    thermal_eff_dict= {}
    weight_factor={}
    for i in range(int(editable_data['number_buildings'])):
        if 'building_name_'+str(i) in editable_data.keys():
            building_name = editable_data['building_name_1']
            idf_names.append(building_name)
            thermal_eff_dict[building_name]=float(editable_data['thermal_eff_'+str(i)])
            weight_factor[building_name]=float(editable_data['WF_'+str(i)])

    #idf_names=idf_names[1:2]
    start_year = int(editable_data['starting_year'])
    end_year = int(editable_data['ending_year'])
    epw_names = []
    for i_temp in range(num_scenarios):
        for i_solar in range(num_scenarios):
            epw_names.append('T_'+str(i_temp)+'_S_'+str(i_solar))
    demand_directory = os.path.join(path_test, 'IDFBuildingsFiles')
    output_directory = os.path.join(path_test, 'IDFBuildingsFiles')

    # epw  main files
    dict_EPWs = {}
    list_years = []
    list_tmys =[]
    list_fmys = []
    for year in range(start_year,end_year+1):
        weather_data = city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)
        list_years.append(weather_data)
    for i in range(5):
        if 'TMY'+str(i+1)+'_name' in editable_data.keys():
            TMY_name = editable_data['TMY'+str(i+1)+'_name']
            list_tmys.append(TMY_name)
        if 'FMY'+str(i+1)+'_name'  in editable_data.keys():
            FMY_name = editable_data['FMY'+str(i+1)+'_name']
            list_fmys.append(FMY_name)
    dict_EPWs['AMYs']=list_years
    dict_EPWs['FMYs']=list_fmys
    dict_EPWs['TMYs']=list_tmys

    global k
    def scenario_reduction_per_year(scenario_genrated,name,weather_data):
        global k
        days= 365
        features_scenarios = defaultdict(list)
        features_scenarios_list = []
        features_probability_list = []
        features_scenarios_nested = nested_dict()
        scenario_probability = [1]*365
        k = 0
        #print(scenario_genrated)
        for i in range(days):
            data_new = scenario_genrated[i*24:(i+1)*24]
            #print(data_new.keys())
            data_1 = data_new['Total Electricity']
            data_2 = data_new['Total Heating']
            #print(data_1)
            #print(name,i,k,data_1[15],data_2[15])
            daily_list =list(chain(data_1.astype('float', copy=False),data_2.astype('float', copy=False)))
            features_scenarios[k*days+i] = daily_list
            features_scenarios_nested[i] = features_scenarios[k*days+i]
            features_scenarios_list.append(features_scenarios[k*days+i])
            features_probability_list.append(scenario_probability[i])
            k = k+1
        A = np.asarray(features_scenarios_list)

        #Convert the dictionary of features to Series
        standardization_data = StandardScaler()
        A_scaled = standardization_data.fit_transform(A)
        inertia_list = []
        search_optimum_cluster = editable_data['Search optimum clusters'] # if I want to search for the optimum number of clusters: 1 is yes, 0 is no
        cluster_range = range(2,30,1)
        if search_optimum_cluster=='yes' and name== 'total_'+dict_EPWs['TMYs'][-1]+'_':
            print('Defining the optimum number of clusters: ')
            SMALL_SIZE = 14
            MEDIUM_SIZE = 18
            BIGGER_SIZE = 24
            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.grid'] = False
            plt.rcParams['axes.edgecolor'] = 'black'
            cmap = plt.cm.RdYlGn
            plt.rcParams["figure.figsize"] = (15,10)
            fig, ax = plt.subplots(figsize=(15, 10))
            for cluster_numbers in cluster_range:
                kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=0).fit(A_scaled)
                inertia_list.append(kmedoids.inertia_)
                plt.scatter(cluster_numbers,kmedoids.inertia_)
                print('Cluster number:', cluster_numbers, '  Inertia of the cluster:', int(kmedoids.inertia_))
            ax.set_xlabel('Number of clusters',fontsize=BIGGER_SIZE)
            ax.set_ylabel('Inertia',fontsize=BIGGER_SIZE)
            #ax.set_title('The user should use "Elbow method" to select the number of optimum clusters',fontsize=BIGGER_SIZE)
            ax.plot(list(cluster_range),inertia_list)
            ax.set_xticks(np.arange(2,30,1))
            plt.savefig(os.path.join(path_test, 'Inertia vs Clusters.png'),dpi=300,facecolor='w')
            plt.close()
            print('"Inertia vs Clusters" figure is saved in the directory folder')
            print('You can use the figure to select the optimum number of clusters' )
            print('You should enter the new optimum number of clusters in EditableFile.csv file and re-run this part')
        cluster_numbers= int(editable_data['Cluster numbers'])
        kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=4).fit(A_scaled)
        #kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=4).fit(scores_pca)
        label = kmedoids.fit_predict(A_scaled)
        #filter rows of original data
        probability_label = defaultdict(list)
        index_label = defaultdict(list)
        index_label_all = []
        filtered_label={}
        for i in range(cluster_numbers):
            filtered_label[i] = A_scaled[label == i]
            index_cluster=np.where(label==i)
            if len(filtered_label[i])!=0:
                index_cluster = index_cluster[0]
                for j in index_cluster:
                    probability_label[i].append(features_probability_list[j])
                    index_label[i].append(j)
                    index_label_all.append(j)
            else:
                probability_label[i].append(0)
        sum_probability = []
        for key in probability_label.keys():
            sum_probability.append(sum(probability_label[key]))


        #print(kmedoids.predict([[0,0,0], [4,4,4]]))
        #print(kmedoids.cluster_centers_,kmedoids.cluster_centers_[0],len(kmedoids.cluster_centers_))
        A_scaled_list={}
        clusters={}
        clusters_list = []
        label_list = []
        data_labels={}
        data_all_labels = defaultdict(list)
        for center in range(len(kmedoids.cluster_centers_)):
            clusters['cluster centers '+str(center)]= kmedoids.cluster_centers_[center]
            clusters_list.append(kmedoids.cluster_centers_[center].tolist())
        for scenario in range(len(A_scaled)):
            data_all_labels[kmedoids.labels_[scenario]].append(standardization_data.inverse_transform(A_scaled[scenario].reshape(1,-1)))
            #print(data_all_labels)
            A_scaled_list[scenario]=A_scaled[scenario].tolist()
            A_scaled_list[scenario].insert(0,kmedoids.labels_[scenario])
            data_labels['labels '+str(scenario)]= A_scaled_list[scenario]
            label_list.append(A_scaled[scenario].tolist())
        df_clusters= pd.DataFrame(clusters)
        df_labels = pd.DataFrame(data_labels)
        df_clusters.to_csv(os.path.join(representative_days_path , name+ 'cluster_centers_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)
        df_labels.to_csv(os.path.join(representative_days_path , name + 'labels_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)
        #Reversing PCA using two methods:
        #Reversing the cluster centers using method 1 (their results are the same)
        Scenario_generated_new = standardization_data.inverse_transform(kmedoids.cluster_centers_)

        #print('15 representative days',clusters_reverse[0][0],Scenario_generated_new[0][0],standardization_data.mean_[0],standardization_data.var_[0])
        representative_day_all = {}
        total_labels = []
        represent_gaps = {}
        scenario_data = {}
        for key in filtered_label.keys():
            total_labels.append(len(filtered_label[key]))
        #print(len(probability_label[0])) 1990
        #print(len(filtered_label[0])) 1990
        for representative_day in range(len(Scenario_generated_new)):
            represent_gaps = {}
            scenario_data = {}
            for i in range(48):
                if Scenario_generated_new[representative_day][i]<0:
                    Scenario_generated_new[representative_day][i] = 0
            for k in range(2): # 2 uncertain inputs
                scenario_data[k] = Scenario_generated_new[representative_day][24*k:24*(k+1)].copy()
                #min_non_z = np.min(np.nonzero(scenario_data[k]))
                #max_non_z = np.max(np.nonzero(scenario_data[k]))
                #represent_gaps[k]= [i for i, x in enumerate(scenario_data[k][min_non_z:max_non_z+1]) if x == 0]
                #ranges = sum((list(t) for t in zip(represent_gaps[k], represent_gaps[k][1:]) if t[0]+1 != t[1]), [])
                #iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
                #print('Present gaps are: ', representative_day,k, 'gaps', ', '.join([str(n) + '-' + str(next(iranges)) for n in iranges]))
                #iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
                #for n in iranges:
                #    next_n = next(iranges)
                #    if (next_n-n) == 0: #for data gaps of 1 hour, get the average value
                #        scenario_data[k][n+min_non_z] = (scenario_data[k][min_non_z+n+1]+scenario_data[k][min_non_z+n-1])/2
                #    elif (next_n-n) > 0  and (next_n-n) <= 6: #for data gaps of 1 hour to 4 hr, use interpolation and extrapolation
                #        f_interpol_short= interp1d([n-1,next_n+1], [scenario_data[k][min_non_z+n-1],scenario_data[k][min_non_z+next_n+1]])
                #        for m in range(n,next_n+1):
                #            scenario_data[k][m+min_non_z] = f_interpol_short(m)
            data_represent_days_modified={'Electricity total (kWh)': scenario_data[0],
            'Heating (kWh)': scenario_data[1],
            'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
            #print(np.mean(Scenario_generated_new[representative_day][0:24]))
            df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
            df_represent_days_modified.to_csv(os.path.join(representative_days_path,name+'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)
        max_heating_scenarios_nested = nested_dict()
        max_electricity_scenarios_nested = nested_dict()
        total_heating_scenarios = []
        total_electricity_scenarios = []
        max_electricity_scenarios_nested_list = defaultdict(list)
        max_heating_scenarios_nested_list = defaultdict(list)
        accuracy_design_day = 0.99
        design_day_heating = []
        design_day_electricity = []
        representative_day_max = {}
        electricity_design_day = {}
        heating_design_day = {}
        for day in range(365):
            for i in range(24):
                k_elect=0
                list_k_electricity = []
                k_heat=0
                list_k_heating = []
                for represent in range(cluster_numbers):
                    representative_day_max[represent] = pd.read_csv(os.path.join(representative_days_path ,name+'Represent_days_modified_'+str(represent)+'.csv'))
                    electricity_demand = representative_day_max[represent]['Electricity total (kWh)'] #kWh
                    heating_demand = representative_day_max[represent]['Heating (kWh)'] #kWh
                    if features_scenarios_nested[day][0:24][i]>electricity_demand[i]:
                        k_elect=1
                        list_k_electricity.append(k_elect)
                    k_elect=0
                    if features_scenarios_nested[day][24:48][i]>heating_demand[i]:
                        k_heat=1
                        list_k_heating.append(k_heat)
                    k_heat=0
                if sum(list_k_electricity)==cluster_numbers: #This hour does not meet by any of the representative days
                    max_electricity_scenarios_nested_list[i].append(features_scenarios_nested[day][0:24][i])
                    total_electricity_scenarios.append(features_scenarios_nested[day][0:24][i])
                if sum(list_k_heating)==cluster_numbers: #This hour does not meet by any of the representative days
                    max_heating_scenarios_nested_list[i].append(features_scenarios_nested[day][24:48][i])
                    total_heating_scenarios.append(features_scenarios_nested[day][24:48][i])
        total_electricity_scenarios.sort(reverse=True)
        total_heating_scenarios.sort(reverse=True)

        max_electricity_hour = total_electricity_scenarios[35]
        max_heating_hour = total_heating_scenarios[2]
        #print(max_heating_hour,len(total_heating_scenarios),np.min(total_heating_scenarios),np.max(total_heating_scenarios))
        design_day_heating = []
        design_day_electricity = []
        heating_dd = []
        for i in range(24):
            if len(max_electricity_scenarios_nested_list[i])==1:
                design_day_electricity.append(max_electricity_scenarios_nested_list[i][0])
            else:
                try:
                    design_day_electricity.append(np.max([j for j in max_electricity_scenarios_nested_list[i] if j<max_electricity_hour]))
                except:
                    design_day_electricity.append(0)
            #print(i,len(max_heating_scenarios_nested_list[i]),max_heating_scenarios_nested_list[i])

            if len(max_heating_scenarios_nested_list[i])==1:
                heating_dd.append(max_heating_scenarios_nested_list[i][0])
                design_day_heating.append(np.max(heating_dd))
            else:
                try:
                    heating_dd = [j for j in max_heating_scenarios_nested_list[i] if j<max_heating_hour]
                    design_day_heating.append(np.max(heating_dd))
                except:
                    design_day_heating.append(0)

        for i in range(24):
            if  design_day_electricity[i]==0:
                if i==0:
                    design_day_electricity[i] = design_day_electricity[i+1]
                elif i==23:
                    design_day_electricity[i] = design_day_electricity[i-1]
                else:
                    design_day_electricity[i] = (design_day_electricity[i-1]+design_day_electricity[i+1])/2
            if  design_day_heating[i]==0:
                if i==0:
                    design_day_heating[i] = design_day_heating[i+1]
                elif i==23:
                    design_day_heating[i] = design_day_heating[i-1]
                else:
                    design_day_heating[i] = (design_day_heating[i-1]+design_day_heating[i+1])/2
        representative_day_max = {}
        electricity_demand_total = defaultdict(list)
        heating_demand_total = defaultdict(list)
        heating_demand_max = {}
        electricity_demand_max = {}
        for represent in range(cluster_numbers):
            representative_day_max[represent] = pd.read_csv(os.path.join(representative_days_path ,name+'Represent_days_modified_'+str(represent)+'.csv'))
            electricity_demand = representative_day_max[represent]['Electricity total (kWh)'] #kWh
            heating_demand = representative_day_max[represent]['Heating (kWh)'] #kWh
            #hours_representative_day= round(sum_probability[representative_day]/sum(sum_probability),4)*8760
            heating_demand_max[represent]= np.mean(heating_demand)
            electricity_demand_max[represent]= np.mean(electricity_demand)
        high_electricity_index = []
        high_heating_index = []
        high_electricity_value = []
        high_heating_value = []
        key_max_electricity=max(electricity_demand_max, key=electricity_demand_max.get)
        key_max_heating=max(heating_demand_max, key=heating_demand_max.get)
        for key, value in max_electricity_scenarios_nested.items():
            for inner_key, inner_value in max_electricity_scenarios_nested[key].items():
                if inner_value>electricity_demand_max[key_max_electricity]:
                    high_electricity_index.append(scenario_number[key]*365+inner_key)
                    high_electricity_value.append(inner_value)
        for key, value in max_heating_scenarios_nested.items():
            for inner_key, inner_value in max_heating_scenarios_nested[key].items():
                if inner_value>heating_demand_max[key_max_heating]:
                    high_heating_index.append(scenario_number[key]*365+inner_key)
                    high_heating_value.append(inner_value)
        sum_probability.append(0.5*len(total_electricity_scenarios)/len(index_label_all)*365)
        sum_probability.append(len(total_heating_scenarios)/len(index_label_all)*365)
        filtered_label[cluster_numbers]=len(total_electricity_scenarios)
        filtered_label[cluster_numbers+1]=len(total_heating_scenarios)
        representative_day = cluster_numbers
        data_represent_days_modified={'Electricity total (kWh)': design_day_electricity,
        'Heating (kWh)': representative_day_max[key_max_electricity]['Heating (kWh)'],
        'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
        df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
        df_represent_days_modified.to_csv(os.path.join(representative_days_path,name+'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)

        representative_day = cluster_numbers+1
        data_represent_days_modified={'Electricity total (kWh)': representative_day_max[key_max_heating]['Electricity total (kWh)'],
        'Heating (kWh)': design_day_heating,
        'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
        df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
        df_represent_days_modified.to_csv(os.path.join(representative_days_path,name+'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)
        for representative_day in range(len(Scenario_generated_new)):
            represent_gaps = {}
            scenario_data = {}
            for i in range(48): #24*5=120 features in each day
                if Scenario_generated_new[representative_day][i]<0:
                    Scenario_generated_new[representative_day][i] = 0
            for k in range(2): # 2 uncertain inputs
                scenario_data[k] = Scenario_generated_new[representative_day][24*k:24*(k+1)].copy()
                #min_non_z = np.min(np.nonzero(scenario_data[k]))
                #zmax_non_z = np.max(np.nonzero(scenario_data[k]))
                #represent_gaps[k]= [i for i, x in enumerate(scenario_data[k][min_non_z:max_non_z+1]) if x == 0]
                #ranges = sum((list(t) for t in zip(represent_gaps[k], represent_gaps[k][1:]) if t[0]+1 != t[1]), [])
                #iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
                #print('Present gaps are: ', representative_day,k, 'gaps', ', '.join([str(n) + '-' + str(next(iranges)) for n in iranges]))
                #iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
                #for n in iranges:
                #    next_n = next(iranges)
                #    if (next_n-n) == 0: #for data gaps of 1 hour, get the average value
                #        scenario_data[k][n+min_non_z] = (scenario_data[k][min_non_z+n+1]+scenario_data[k][min_non_z+n-1])/2
                #    elif (next_n-n) > 0  and (next_n-n) <= 6: #for data gaps of 1 hour to 4 hr, use interpolation and extrapolation
                #        f_interpol_short= interp1d([n-1,next_n+1], [scenario_data[k][min_non_z+n-1],scenario_data[k][min_non_z+next_n+1]])
                #        for m in range(n,next_n+1):
                #            scenario_data[k][m+min_non_z] = f_interpol_short(m)
            data_represent_days_modified={'Electricity total (kWh)': scenario_data[0],
            'Heating (kWh)': scenario_data[1],
            'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
            #print(np.mean(Scenario_generated_new[representative_day][0:24]))
            df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
            df_represent_days_modified.to_csv(os.path.join(representative_days_path,name + 'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)
        all_representative_days = clustring_kmean_forced.kmedoid_clusters(path_test,scenario_genrated,name)[2]
        represent_day = defaultdict(list)
        k=0
        days= 365

        for represent in range(int(editable_data['Cluster numbers'])+2):
            for day in range(days):
                data = scenario_genrated[day*24:(day+1)*24]
                data_1 = data['Total Electricity']
                data_2 = data['Total Heating']
                #Total electricity and heating
                daily_list =list(chain(data_1.astype('float', copy=False),data_2.astype('float', copy=False)))
                #if round(all_representative_days[represent]['Electricity total (kWh)'][10],0)==round(daily_list[10],0):
                #    print('elect',represent, day, round(all_representative_days[represent]['Electricity total (kWh)'][10],0),round(daily_list[10],0))
                #if round(all_representative_days[represent]['Heating (kWh)'][6],0)==round(daily_list[30],0):
                #    print('heat',represent, day, round(all_representative_days[represent]['Heating (kWh)'][6],0),round(daily_list[30],0))
                if round(all_representative_days[represent]['Electricity total (kWh)'][10],0)==round(daily_list[10],0) and round(all_representative_days[represent]['Heating (kWh)'][6],0)==round(daily_list[30],0) :
                    represent_day[represent] = day
                    data_temp = []
                    data_dni = []
                    data_ghi = []
                    data_dhi = []
                    data_wind_speed = []
                    poa_components_vector = []
                    poa_global = []
                    hour = 0
                    for index_in_year in range(day*24,(day+1)*24):
                        data_temp.append(weather_data['temp_air'].tolist()[index_in_year])
                        data_dni.append(weather_data['dni'].tolist()[index_in_year])
                        data_ghi.append(weather_data['ghi'].tolist()[index_in_year])
                        data_dhi.append(weather_data['dhi'].tolist()[index_in_year])
                        data_wind_speed.append(weather_data['wind_speed'].tolist()[index_in_year])
                        dti =  datetime.datetime(weather_data['year'].tolist()[index_in_year], weather_data['month'].tolist()[index_in_year], weather_data['day'].tolist()[index_in_year],hour)
                        solar_position = get_solarposition(dti,lat, lon, altitude, pressure=None, method='nrel_numpy', temperature=12)
                        solar_zenith = solar_position['zenith']
                        solar_azimuth =  solar_position['azimuth']
                        poa_components_vector.append(get_total_irradiance(surf_tilt, surf_azimuth,
                                                     solar_zenith[0], solar_azimuth[0],
                                                    float(weather_data['dni'].tolist()[index_in_year]), float(weather_data['ghi'].tolist()[index_in_year]), float(weather_data['dhi'].tolist()[index_in_year]), dni_extra=None, airmass=None,
                                                     albedo=.25, surface_type=None,
                                                     model='isotropic',
                                                     model_perez='allsitescomposite1990'))
                        poa_global.append(poa_components_vector[hour]['poa_global'])
                        hour +=1
                    for represent in range(int(editable_data['Cluster numbers'])+2):
                        all_representative_days[represent]['temp_air']=data_temp
                        all_representative_days[represent]['dni']=data_dni
                        all_representative_days[represent]['ghi']=data_ghi
                        all_representative_days[represent]['dhi']=data_dhi
                        all_representative_days[represent]['wind_speed']=data_wind_speed
                        all_representative_days[represent]['gti']=poa_global
                        all_representative_days[represent].to_csv(os.path.join(representative_days_path,name + 'Represent_days_modified_'+str(represent)+ '.csv'), index=False)
                    break
        return data_all_labels, represent_day

    cluster_numbers= int(editable_data['Cluster numbers'])+2
    temps= []
    gtis=[]
    for scenario in range(len(epw_names)):
        #output_prefix =  building_type+'_'+epw_names[scenario]+'_'
        weather_path = os.path.join(scenarios_path,epw_names[scenario]+'.csv')
        data = pd.read_csv(weather_path)
        if scenario<10:
            gtis.append(round(np.mean(data['GTI']),1))
            #print(epw_names[scenario],'GTI',np.mean(data['GTI']))
        if scenario%10==0:
            #print(epw_names[scenario],'Temp',np.mean(data['Temperature']))
            temps.append(round(np.mean(data['Temperature']),1))
    print('gti', gtis)
    print('temps',temps)

    scenario_generated_main = defaultdict(list)
    elect_buildings_main = defaultdict(list)
    gas_buildings_main = defaultdict(list)
    elect_annual_main = defaultdict(list)
    gas_annual_main = defaultdict(list)
    for building_type in idf_names:
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  building_type+'_'+epw_file_name+'_mtr.csv'
                demand_data_path = os.path.join(demand_directory, output_prefix)
                data = pd.read_csv(demand_data_path)
                elect_data = ((data['Electricity:Facility [J](Hourly)']-data['Heating:Electricity [J](Hourly)'])*JtokWh)
                heat_data = (data['Gas:Facility [J](Hourly)']*thermal_eff_dict[building_type]+data['Heating:Electricity [J](Hourly)'])*JtokWh
                #data['Total Electricity']=elect_data
                #data['Total Heating']=heat_data
                scenario_generated_main[building_type].append(data)
                elect_buildings_main[building_type].append(elect_data)
                elect_annual_main[building_type].append(sum(elect_data))
                gas_buildings_main[building_type].append(heat_data)
                gas_annual_main[building_type].append(sum(heat_data))
    j=0
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            if key =='AMYs':
                weather_path = os.path.join(scenarios_path,epw_file_name+'.epw')
                data, meta = EPW_to_csv.read_epw(weather_path)
            elif key =='FMYs':
                weather_path = os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.epw')
                data, meta = EPW_to_csv.read_epw(weather_path,FMYs='yes')
            else:
                weather_path = os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.epw')
                data, meta = EPW_to_csv.read_epw(weather_path)
            data.to_csv(os.path.join(scenarios_path,epw_file_name+'.csv'), index = False, header=True)
            total_electricity_buildings = []
            total_heating_buildings = []
            for building_type in idf_names:
                if mode=='seperate':
                    output_prefix =  building_type+'_'+epw_file_name+'_'
                    scenario_generated_main[building_type][j]['Total Electricity']=elect_buildings_main[building_type][j]*weight_factor[building_type]
                    scenario_generated_main[building_type][j]['Total Heating']=gas_buildings_main[building_type][j]*weight_factor[building_type]
                    scenario_reduction_per_year(scenario_generated_main[building_type][j],output_prefix,data)
                elif mode=='total':
                    total_electricity_buildings.append(elect_buildings_main[building_type][j]*weight_factor[building_type])
                    total_heating_buildings.append(gas_buildings_main[building_type][j]*weight_factor[building_type])
            if mode=='total':
                output_prefix =  'total_'+epw_file_name+'_'
                scenario_generated_main[building_type][j]['Total Electricity']=sum(total_electricity_buildings)
                scenario_generated_main[building_type][j]['Total Heating']=sum(total_heating_buildings)
                #print(total_electricity_buildings[0][15],total_electricity_buildings[1][15],total_electricity_buildings[2][15],sum(total_electricity_buildings)[15],len(sum(total_electricity_buildings)))
                #print(len(scenario_generated_main[building_type][j]))
                scenario_reduction_per_year(scenario_generated_main[building_type][j],output_prefix,data)
            j = j+1


    scenario_probability = defaultdict(list)
    scenario_generated = defaultdict(list)
    elect_buildings = defaultdict(list)
    gas_buildings = defaultdict(list)
    elect_annual= defaultdict(list)
    gas_annual = defaultdict(list)
    for building_type in idf_names:
        for scenario in range(len(epw_names)):
            output_prefix =  building_type+'_'+epw_names[scenario]+'_mtr.csv'
            demand_data_path = os.path.join(demand_directory, output_prefix)
            data = pd.read_csv(demand_data_path)
            elect_data = (data['Electricity:Facility [J](Hourly)']-data['Heating:Electricity [J](Hourly)'])*JtokWh
            heat_data = (data['Gas:Facility [J](Hourly)']*thermal_eff_dict[building_type]+data['Heating:Electricity [J](Hourly)'])*JtokWh
            #data['Total Electricity']=elect_data
            #data['Total Heating']=heat_data
            scenario_generated[building_type].append(data)
            scenario_generated[building_type].append(data)
            elect_buildings[building_type].append(elect_data)
            elect_annual[building_type].append(sum(elect_data))
            gas_buildings[building_type].append(heat_data)
            gas_annual[building_type].append(sum(heat_data))
            #print(scenario,output_prefix,gas_buildings[building_type][scenario][0],elect_buildings[building_type][scenario][0])
    for scenario in range(len(epw_names)):
        output_prefix =  building_type+'_'+epw_names[scenario]+'_'
        weather_path = os.path.join(scenarios_path,epw_names[scenario]+'.epw')
        data, meta = EPW_to_csv.read_epw(weather_path)
        data.to_csv(os.path.join(scenarios_path,epw_file_name+'.csv'), index = False, header=True)
        total_electricity_buildings = []
        total_heating_buildings = []
        for building_type in idf_names:
            if mode=='seperate':
                output_prefix =  building_type+'_'+epw_names[scenario]+'_'
                scenario_generated[building_type][scenario]['Total Electricity']=elect_buildings[building_type][scenario]*weight_factor[building_type]
                scenario_generated[building_type][scenario]['Total Heating']=gas_buildings[building_type][scenario]*weight_factor[building_type]
                scenario_reduction_per_year(scenario_generated[building_type][scenario],output_prefix,data)
            elif mode=='total':
                total_electricity_buildings.append(elect_buildings[building_type][scenario]*weight_factor[building_type])
                total_heating_buildings.append(gas_buildings[building_type][scenario]*weight_factor[building_type])
        if mode=='total':
            output_prefix =  'total_'+epw_names[scenario]+'_'
            scenario_generated[building_type][scenario]['Total Electricity']=sum(total_electricity_buildings)
            scenario_generated[building_type][scenario]['Total Heating']=sum(total_heating_buildings)
            #print(scenario_generated[building_type][scenario].keys())
            scenario_reduction_per_year(scenario_generated[building_type][scenario],output_prefix,data)
