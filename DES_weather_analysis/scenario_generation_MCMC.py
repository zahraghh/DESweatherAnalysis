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
    #weather_keys = weather_keys[:2]
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
        for day in range(num_days):
            print('scenario', scenario,'day', day)
            for weather_input in weather_keys: #getiing the sequential weather parameters over a day
                rand_selected=random.random() #initialize the acceptence probability
                for j in range(100):#change to 100 #number of iterations in the whole 24 hours.
                    selected = []
                    selected_acceptence = []
                    for hour in range(num_hours):
                        #print('hour', hour)
                        index = day*num_hours+hour
                        scenario_selected=random.randint(0, num_scenarios-1)
                        #print(weather_input, hour,len(weather_scenario[weather_input][index]),scenario_selected,weather_scenario[weather_input][index])
                        if (hour==0) or (round(np.mean(weather_scenario[weather_input][index][scenario_selected]),3)==0):
                            if weather_input!='temp_air':
                                if round(weather_scenario[weather_input][index][scenario_selected],3)<0:
                                    weather_scenario[weather_input][index][scenario_selected]=0
                            selected.append(round(weather_scenario[weather_input][index][scenario_selected],3))
                            accept_selected = rand_selected
                            #print('day-scenario',day, scenario,'gti hour',hour,'value ', s2[hour])
                        else:
                            while(len(selected)==hour): #while a transition is not accepted
                                list_states_0 = [round(abs(i-selected[hour-1]),3) for i in list(UA_weather_data[weather_input][str(day*num_hours+hour-1)])]
                                index_pre = list_states_0.index(min(list_states_0))
                                list_states_1 = [round(abs(i-UA_weather_data[weather_input][str(day*num_hours+hour)][index_pre]),3) for i in weather_scenario[weather_input][index]]
                                if sum(list_states_1)==0:
                                    #print('inja',weather_input,UA_weather_data[weather_input][str(day*num_hours+hour)][index_pre],weather_scenario[weather_input][index])
                                    selected.append(round(weather_scenario[weather_input][index][scenario_selected],3))
                                else:
                                    list_states_1 = [z/sum(list_states_1) for z in list_states_1]
                                    scenario_selected=random.randint(0, len(list_states_1)-1)
                                    if (accept_selected<(1-list_states_1[scenario_selected])):
                                        #Accepting the transition
                                        selected.append(round(weather_scenario[weather_input][index][scenario_selected],3))
                                        selected_acceptence.append(1-list_states_1[scenario_selected])
                                        accept_selected = 1-list_states_1[scenario_selected]
                                    else:
                                        #Rejecting the transition
                                        rand_selected=round(random.uniform(0,1),5)
                                        accept_selected = rand_selected #changing the acceptence theta
                                #print(weather_input,selected)
                    if j==0:
                        selected_final = selected
                        selected_prob = np.prod(selected_acceptence)
                    else:
                        if  selected_prob < np.prod(selected_acceptence):
                            selected_final = selected
                            selected_prob = np.prod(selected_acceptence)
                    #print(weather_input, selected)
                #if j%10==0:
                    #print(j)
                    #print(weather_input,'probs', selected_prob)
                    #print(weather_input,'final',selected_final)
                #print('total time', (time.time()-timeout_scenario)/60)
                day_final = []
                for hour in range(num_hours):
                    day_final.append(selected_final[hour])
                generated_scenario[weather_input][scenario][day]=day_final
                #print('end state',day_final)

    #print('1',len(generated_scenario['gti']),generated_scenario['gti'])
    #print('2',len(generated_scenario['gti'][scenario]),generated_scenario['gti'][scenario])
    generated_scenario_sorted = defaultdict(lambda: defaultdict(list))
    for weather_input in weather_keys:
        for day in range(num_days):
            mean_data = {}
            for scenario in range(num_scenarios):
                mean_data[scenario] = np.mean(generated_scenario[weather_input][scenario][day])
                #print('here mean',scenario,np.mean(generated_scenario[weather_input][scenario][day]))
            sort_keys={k: v for k, v in sorted(mean_data.items(), key=lambda item: item[1])}
            #print('keys',sort_keys.keys())
            j=0
            for key in sort_keys.keys():
                generated_scenario_sorted[weather_input][j].append(generated_scenario[weather_input][key][day])
                j = j+1
                #print('here mean',key, scenario,np.mean(generated_scenario[weather_input][key][day]))
            #print('final',day, len(generated_scenario_sorted[weather_input]))

    for weather_input in weather_keys:
        for scenario in range(num_scenarios):
            generated_scenario_sorted[weather_input][scenario] = list(itertools.chain.from_iterable(generated_scenario_sorted[weather_input][scenario]))
            print(generated_scenario_sorted[weather_input][scenario])
    scenario_genrated = {}
    scenario_genrated_normalized = {}
    for i_temp in range(int(num_scenarios/2)):
        for i_solar in range(int(num_scenarios/2)):
                print(i_temp, i_solar)
                scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)] = {'Temperature':generated_scenario_sorted['temp_air'][i_temp*2],
                 'GHI':generated_scenario_sorted['ghi'][i_solar*2],
                 'DNI':generated_scenario_sorted['dni'][i_solar*2],
                 'DHI':generated_scenario_sorted['dhi'][i_solar*2],
                 'GTI':generated_scenario_sorted['gti'][i_solar*2]}
                df_scenario_generated=pd.DataFrame(scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)])
                #print('finalfinal',df_scenario_generated)
                df_scenario_generated.to_csv(os.path.join(save_path,'T_'+str(i_temp)+'_S_'+str(i_solar)+'.csv'), index=False)

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
