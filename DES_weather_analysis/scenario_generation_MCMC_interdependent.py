import numpy as np
import pandas as pd
import csv
import random
import itertools
from collections import defaultdict
from nested_dict import nested_dict
from scipy import stats
import os
import sys
import json
import pyDOE
import scipy.stats as st
import datetime
from copulas.datasets import sample_bivariate_age_income
from copulas.multivariate import GaussianMultivariate
import DES_weather_analysis
from DES_weather_analysis import weather_file_PDFs
from DES_weather_analysis import solar_irradiance
from DES_weather_analysis.solar_irradiance import aoi, get_total_irradiance
from DES_weather_analysis.solar_position import get_solarposition
from pvlib import atmosphere, solarposition, tools

def scenario_generation_results(path_test):
    lbstokg_convert = 0.453592 #1 lb = 0.453592 kg
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    weather_keys = ['gti','temp_air','dhi','dni','ghi']
    weather_keys = ['ghi', 'temp_air']
    num_scenarios = int(editable_data['num_scenarios'])*2
    #num_scenarios_LHS = int(editable_data['num_scenarios_intervals'])
    ending_year = editable_data['ending_year']
    start_year = 1998
    end_year = 2019
    city=editable_data['city']
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    folder_path = os.path.join(path_test,str(city))
    dict_weather_csv = {}
    dict_EPWs = {}
    list_years = []
    list_tmys =[]
    list_fmys = []
    for year in range(start_year,end_year+1):
        weather_data = city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)
        list_years.append(weather_data)
    dict_EPWs['AMYs']=list_years

    #Only reading the weather files
    temp_air = []
    gti = []
    ghi = []
    dni = []
    dhi = []
    wind_speed = []

    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            if key=='AMYs':
                dict_weather_csv[epw_file_name] =   pd.read_csv(os.path.join(folder_path,epw_file_name+'.csv'), header=2)[0:8760]
                weather_path = os.path.join(folder_path,epw_file_name+'.csv')
                temp_air.append(dict_weather_csv[epw_file_name]['Temperature'])
                ghi.append(dict_weather_csv[epw_file_name]['GHI'])
                dni.append(dict_weather_csv[epw_file_name]['DNI'])
                dhi.append(dict_weather_csv[epw_file_name]['DHI'])
                wind_speed.append(dict_weather_csv[epw_file_name]['Wind Speed'])

    #weather_params={'ghi':ghi, 'dni':dni,'dhi':dhi,'temp_air':temp_air, 'wind_speed':wind_speed}
    weather_params={'ghi':ghi, 'dni':dni,'dhi':dhi,'temp_air':temp_air}
    save_path = os.path.join(path_test,'ScenarioGeneration')
    num_hours= 24
    num_days=365
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    UA_weather_data={}
    generated_scenario = nested_dict(3,list)
    for weather_input in weather_keys:
        try:
            f=open(os.path.join(path_test, 'UA_'+weather_input+'.json'))
            UA_weather_data[weather_input] = json.load(f)
        except IOError:
            print('Please, run weather_file_PDFs first')
    weather_scenario = defaultdict(dict)
    weather_scenario = defaultdict(dict)
    for i in range(num_days*num_hours):
        weather_bivariate = pd.DataFrame({'ghi':UA_weather_data['ghi'][str(i)],'temp_air':UA_weather_data['temp_air'][str(i)]})
        copula = GaussianMultivariate()
        copula.fit(weather_bivariate)
        weather_scenario[i] = copula.sample(num_scenarios)

    for scenario in range(num_scenarios):
        print('scenario', scenario)

        for day in range(num_days):
            rand_selected=random.random() #initialize the acceptence probability
            for j in range(100):#change to 100 #number of iterations in the whole 24 hours.
                selected = defaultdict(list)
                selected_acceptence = defaultdict(list)
                for hour in range(num_hours):
                    #print('hour', hour)
                    index = day*num_hours+hour
                    scenario_selected=random.randint(0, num_scenarios-1)
                    #print(weather_input, hour,len(weather_scenario[weather_input][index]),scenario_selected,weather_scenario[weather_input][index])
                    if (hour==0) or (round(np.mean(weather_scenario[index]['ghi'][scenario_selected]),3)==0):
                        for weather_input in weather_keys: #getiing the sequential weather parameters over a day
                            if round(weather_scenario[index]['ghi'][scenario_selected],3)<0:
                                weather_scenario[index]['ghi'][scenario_selected]=0
                            selected[weather_input].append(round(weather_scenario[index][weather_input][scenario_selected],3))
                            accept_selected= rand_selected
                    else:
                        while(len(selected['ghi'])==hour): #while a transition is not accepted
                            list_states_0_ghi = [round(abs(i-selected['ghi'][hour-1]),3) for i in list(UA_weather_data['ghi'][str(day*num_hours+hour-1)])]
                            list_states_0_temp = [round(abs(i-selected['temp_air'][hour-1]),3) for i in list(UA_weather_data['temp_air'][str(day*num_hours+hour-1)])]
                            list_states_0 = [(x**2+y**2)**0.5 for (x,y) in zip(list_states_0_ghi,list_states_0_temp)]
                            index_pre = list_states_0.index(min(list_states_0))


                            list_states_1_ghi = [round(abs(i-UA_weather_data['ghi'][str(day*num_hours+hour)][index_pre]),3) for i in weather_scenario[index]['ghi']]
                            list_states_1_temp = [round(abs(i-UA_weather_data['temp_air'][str(day*num_hours+hour)][index_pre]),3) for i in weather_scenario[index]['temp_air']]
                            list_states_1 = [(x**2+y**2)**0.5 for (x,y) in zip(list_states_1_ghi,list_states_1_temp)]

                            if sum(list_states_1)==0:
                                #print('inja',weather_input,UA_weather_data[weather_input][str(day*num_hours+hour)][index_pre],weather_scenario[weather_input][index])
                                for weather_input in weather_keys: #getiing the sequential weather parameters over a day
                                    selected[weather_input].append(round(weather_scenario[index][weather_input][scenario_selected],3))
                            else:
                                list_states_1_ghi = [z/sum(list_states_1_ghi) for z in list_states_1_ghi]
                                list_states_1_temp = [z/sum(list_states_1_temp) for z in list_states_1_temp]

                                scenario_selected=random.randint(0, len(list_states_1_ghi)-1)
                                if (accept_selected<((1-list_states_1_ghi[scenario_selected])*(1-list_states_1_temp[scenario_selected]))):
                                    #Accepting the transition
                                    selected['ghi'].append(round(weather_scenario[index]['ghi'][scenario_selected],3))
                                    selected['temp_air'].append(round(weather_scenario[index]['temp_air'][scenario_selected],3))
                                    selected_acceptence[weather_input].append(((1-list_states_1_ghi[scenario_selected])*(1-list_states_1_temp[scenario_selected])))
                                    accept_selected= (1-list_states_1_ghi[scenario_selected])*(1-list_states_1_temp[scenario_selected])
                                else:
                                    #Rejecting the transition
                                    rand_selected=round(random.uniform(0,1),5)
                                    accept_selected = rand_selected #changing the acceptence theta
                    #print(hour, selected_final)
                #print('ghi',selected['ghi'],'temp',selected['temp_air'])
                if j==0:
                    selected_final = selected['ghi']+selected['temp_air']
                    selected_prob = np.prod(selected_acceptence['ghi']+selected_acceptence['temp_air'])
                else:
                    if  selected_prob < np.prod(selected_acceptence['ghi']+selected_acceptence['temp_air']):
                        selected_final = selected['ghi']+selected['temp_air']
                        selected_prob = np.prod(selected_acceptence['ghi']+selected_acceptence['temp_air'])
                #if j%20==0:
                #    print(j)
                #    print('probs', selected_prob)
                #    print('final',selected_final)
                #print('total time', (time.time()-timeout_scenario)/60)
            day_final_ghi = []
            day_final_temp = []

            for hour in range(num_hours):
                day_final_ghi.append(selected_final[hour])
                day_final_temp.append(selected_final[hour+num_hours])

            generated_scenario['ghi'][scenario][day]=day_final_ghi
            generated_scenario['temp_air'][scenario][day]=day_final_temp

            #print('end state ghi',day_final_ghi)
            #print('end state temp',day_final_temp)
    #print('1',len(generated_scenario['gti']),generated_scenario['gti'])
    #print('2',len(generated_scenario['gti'][scenario]),generated_scenario['gti'][scenario])
    generated_scenario_sorted = defaultdict(lambda: defaultdict(list))
    for day in range(num_days):
        mean_data = {}
        for scenario in range(num_scenarios):
            mean_data[scenario] = np.mean(generated_scenario['temp_air'][scenario][day])
            #print('here mean',scenario,np.mean(generated_scenario[weather_input][scenario][day]))
        sort_keys={k: v for k, v in sorted(mean_data.items(), key=lambda item: item[1])}
        #print('keys',sort_keys.keys())
        j=0
        for key in sort_keys.keys():
            generated_scenario_sorted['temp_air'][j].append(generated_scenario['temp_air'][key][day])
            generated_scenario_sorted['ghi'][j].append(generated_scenario['ghi'][key][day])

            j = j+1
            #print('here mean',key, scenario,np.mean(generated_scenario['temp_air'][key][day]),np.mean(generated_scenario['ghi'][key][day]))
        #print('final',day, len(generated_scenario_sorted[weather_input]))

    for scenario in range(num_scenarios):
        generated_scenario_sorted['temp_air'][scenario] = list(itertools.chain.from_iterable(generated_scenario_sorted['temp_air'][scenario]))
        generated_scenario_sorted['ghi'][scenario] = list(itertools.chain.from_iterable(generated_scenario_sorted['ghi'][scenario]))
        #print(generated_scenario_sorted['ghi'][scenario])

    scenario_genrated = {}
    scenario_genrated_normalized = {}
    for scenario in range(int(num_scenarios)):
        scenario_genrated['dependent_'+str(scenario)] = {'Temperature':generated_scenario_sorted['temp_air'][scenario],
         'GHI':generated_scenario_sorted['ghi'][scenario]}
         #'DNI':generated_scenario_sorted['dni'][i_solar*2],
         #'DHI':generated_scenario_sorted['dhi'][i_solar*2],
         #'GTI':generated_scenario_sorted['gti'][i_solar*2]}
        df_scenario_generated =pd.DataFrame(scenario_genrated['dependent_'+str(scenario)])
        #print('finalfinal',df_scenario_generated)
        df_scenario_generated.to_csv(os.path.join(save_path,'dependent_'+str(scenario)+'.csv'), index=False)
