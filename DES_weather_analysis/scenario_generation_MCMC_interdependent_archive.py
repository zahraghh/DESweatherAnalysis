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
    weather_keys = ['temp_air','ghi']
    num_scenarios = int(editable_data['num_scenarios'])*2
    #num_scenarios_LHS = int(editable_data['num_scenarios_intervals'])
    ending_year = editable_data['ending_year']
    save_path = os.path.join(path_test,'ScenarioGeneration')
    num_hours= 24
    num_days=365
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #Normal distribution for electricity emissions
    #year = int(editable_data['ending_year']) #Getting the last year of wind and solar data from NSRDB
    #epw_file_name = 'USA_UT_Salt-Lake-City-Intl-AP.725720_AMY_'+str(ending_year)
    weather_distribution = {}
    #weather_data = pd.read_csv(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),'AMYs'),epw_file_name+'.csv'))
    for weather_input in weather_keys:
        with open(os.path.join(path_test,'best_fit_'+weather_input+'.json')) as f:
            weather_distribution[weather_input] = json.load(f)
    weather_scenario = defaultdict(list)
    for weather_input in weather_keys:
        for i in range(num_days*num_hours):
            if weather_distribution[weather_input][i][0] == 'constant':
                weather_scenario[weather_input].append([weather_distribution[weather_input][i][1]]*num_scenarios)
            else:
                dist_i = getattr(st, weather_distribution[weather_input][i][0])
                params = weather_distribution[weather_input][i][1]
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                if (dist_i.name=='norm' or dist_i.name=='uniform' or dist_i.name=='expon' or dist_i.name=='cauchy'):
                    dist_rd_value = dist_i.rvs(loc= params[0] , scale= params[1] , size=num_scenarios)
                elif (dist_i.name=='weibull_min' or dist_i.name=='weibull_max' or dist_i.name=='triang'):
                    dist_rd_value = dist_i.rvs(c=params[0], loc= params[1] , scale= params[2] , size=num_scenarios)
                elif dist_i.name=='gamma':
                    dist_rd_value = dist_i.rvs(a=params[0], loc= params[1] , scale= params[2] , size=num_scenarios)
                elif dist_i.name=='chi':
                    dist_rd_value = dist_i.rvs(df=params[0], loc= params[1] , scale= params[2] , size=num_scenarios)
                elif dist_i.name=='lognorm':
                    dist_rd_value = dist_i.rvs(s=params[0], loc= params[1] , scale= params[2] , size=num_scenarios)
                elif dist_i.name=='beta':
                    dist_rd_value = dist_i.rvs(a=params[0], b=params[1], loc= params[2] , scale= params[3] , size=num_scenarios)
                elif dist_i.name=='f':
                    dist_rd_value = dist_i.rvs(dfn=params[0], dfd=params[1], loc= params[2] , scale= params[3] , size=num_scenarios)
                weather_scenario[weather_input].append(dist_rd_value)

    UA_weather_data={}
    generated_scenario = nested_dict(3,list)
    for weather_input in weather_keys:
        try:
            f=open(os.path.join(path_test, 'UA_'+weather_input+'.json'))
            UA_weather_data[weather_input] = json.load(f)
        except IOError:
            print('Please, run weather_file_PDFs first')

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
                    if (hour==0) or (round(np.mean(weather_scenario['ghi'][index][scenario_selected]),3)==0):
                        for weather_input in weather_keys: #getiing the sequential weather parameters over a day
                            if weather_input!='temp_air':
                                if round(weather_scenario[weather_input][index][scenario_selected],3)<0:
                                    weather_scenario[weather_input][index][scenario_selected]=0
                            selected[weather_input].append(round(weather_scenario[weather_input][index][scenario_selected],3))
                            accept_selected= rand_selected
                    else:
                        while(len(selected['ghi'])==hour): #while a transition is not accepted
                            list_states_0_ghi = [round(abs(i-selected['ghi'][hour-1]),3) for i in list(UA_weather_data['ghi'][str(day*num_hours+hour-1)])]
                            list_states_0_temp = [round(abs(i-selected['temp_air'][hour-1]),3) for i in list(UA_weather_data['temp_air'][str(day*num_hours+hour-1)])]

                            index_pre_ghi = list_states_0_ghi.index(min(list_states_0_ghi))
                            index_pre_temp = list_states_0_temp.index(min(list_states_0_temp))

                            list_states_1_ghi = [round(abs(i-UA_weather_data['ghi'][str(day*num_hours+hour)][index_pre_ghi]),3) for i in weather_scenario['ghi'][index]]
                            list_states_1_temp = [round(abs(i-UA_weather_data['temp_air'][str(day*num_hours+hour)][index_pre_temp]),3) for i in weather_scenario['temp_air'][index]]

                            if sum(list_states_1_ghi)*sum(list_states_1_temp)==0:
                                #print('inja',weather_input,UA_weather_data[weather_input][str(day*num_hours+hour)][index_pre],weather_scenario[weather_input][index])
                                for weather_input in weather_keys: #getiing the sequential weather parameters over a day
                                    selected[weather_input].append(round(weather_scenario[weather_input][index][scenario_selected],3))
                            else:
                                list_states_1_ghi = [z/sum(list_states_1_ghi) for z in list_states_1_ghi]
                                list_states_1_temp = [z/sum(list_states_1_temp) for z in list_states_1_temp]

                                scenario_selected=random.randint(0, len(list_states_1_ghi)-1)
                                if (accept_selected<((1-list_states_1_ghi[scenario_selected])*(1-list_states_1_temp[scenario_selected]))):
                                    #Accepting the transition
                                    selected['ghi'].append(round(weather_scenario['ghi'][index][scenario_selected],3))
                                    selected['temp_air'].append(round(weather_scenario['temp_air'][index][scenario_selected],3))
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
            mean_data[scenario] = np.mean(generated_scenario['ghi'][scenario][day])
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

def scenario_analysis(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    weather_keys = ['GHI','DNI','DHI','GTI']
    #weather_keys = weather_keys[:2]
    num_scenarios = int(editable_data['num_scenarios'])
    scenario_genrated = {}
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    altitude = float(editable_data['Altitude']) #SLC altitude m
    surf_tilt = float(editable_data['solar_tilt']) #panels tilt degree
    surf_azimuth = float(editable_data['solar_azimuth']) #panels azimuth degree
    end_year = int(editable_data['ending_year'])
    for i_temp in range(int(num_scenarios)):
        for i_solar in range(int(num_scenarios)):
            save_path = os.path.join(path_test,'ScenarioGeneration')
            scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)]=pd.read_csv(os.path.join(save_path,'T_'+str(i_temp)+'_S_'+str(i_solar)+'.csv'))
            for key in weather_keys:
                for i in range(8760):
                    if scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)][key][i]!=0 and scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)][key][i]<0:
                        scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)][key][i]=(scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)][key][i-1]+scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)][key][i+1])/2
                    if scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)]['Temperature'][i]<-70 or scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)]['Temperature'][i]>60:
                        scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)]['Temperature'][i]=(scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)][key][i-1]+scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)][key][i+1])/2
            weather_data = scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)]
            DNI= weather_data['DNI']
            DHI = weather_data['DHI']
            GHI = weather_data['GHI']
            dti = pd.date_range(str(end_year)+"-01-01", periods=8760, freq="H")
            solar_position = get_solarposition(dti, lat, lon, altitude, pressure=None, method='nrel_numpy', temperature=12)
            solar_zenith = solar_position['zenith']
            solar_azimuth =  solar_position['azimuth']
            poa_components_vector = []
            poa_global = []
            for i in range(len(solar_zenith)):
                poa_components_vector.append(get_total_irradiance(surf_tilt, surf_azimuth,
                                         solar_zenith[i], solar_azimuth[i],
                                        float(DNI[i]), float(GHI[i]), float(DHI[i]), dni_extra=None, airmass=None,
                                         albedo=.25, surface_type=None,
                                         model='isotropic',
                                         model_perez='allsitescomposite1990'))
                poa_global.append(poa_components_vector[i]['poa_global'])
            scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)]['GTI_calculated'] = poa_global
            df_scenario_generated=pd.DataFrame(scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)])
            df_scenario_generated.to_csv(os.path.join(save_path,'T_'+str(i_temp)+'_S_'+str(i_solar)+'.csv'), index=False)
