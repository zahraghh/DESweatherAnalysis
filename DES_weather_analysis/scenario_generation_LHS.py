import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from scipy import stats
import os
import sys
import json
import pyDOE
import scipy.stats as st

path_test =  os.path.join(sys.path[0])
lbstokg_convert = 0.453592 #1 lb = 0.453592 kg
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
weather_keys = ['temp_air','dhi','dni','ghi','gti']
num_scenarios = int(editable_data['num_scenarios'])
num_scenarios_revised = num_scenarios + 4
def scenario_generation_results(path_test,weather_keys):
    ending_year = editable_data['ending_year']
    save_path = os.path.join(sys.path[0],'ScenarioGeneration')
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
    weather_scenario = defaultdict(lambda: defaultdict(list))
    for weather_input in weather_keys:
        for i in range(8760):
            if weather_distribution[weather_input][i][0] == 'constant':
                for scenario in range(num_scenarios):
                    weather_scenario[weather_input][scenario].append(weather_distribution[weather_input][i][1])
            else:
                lhd = pyDOE.lhs(1, samples=num_scenarios_revised)
                #if len(weather_distribution[weather_input][i][0])==2:
                dist_i = getattr(st, weather_distribution[weather_input][i][0])
                if dist_i.shapes is None:
                    lhd= dist_i(loc=weather_distribution[weather_input][i][1][-2], scale=weather_distribution[weather_input][i][1][-1]).ppf(lhd)  # this applies to both factors here
                else:
                    params = weather_distribution[weather_input][i][1]
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                     #DISTRIBUTIONS = [st.beta,st.norm, st.uniform, st.expon,
                     #st.weibull_min,st.weibull_max,st.gamma,st.chi,st.lognorm,st.cauchy,st.triang,st.f]
                    if dist_i.name=='weibull_max' or dist_i.name=='weibull_min' or dist_i.name=='triang':
                        lhd= dist_i(c=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
                    elif dist_i.name=='beta':
                        lhd= dist_i(a=arg[0],b=arg[1],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
                    elif dist_i.name=='gamma':
                        lhd= dist_i(a=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
                    elif dist_i.name=='lognorm':
                        lhd= dist_i(s=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
                    elif dist_i.name=='chi':
                        lhd= dist_i(df=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
                    elif dist_i.name=='f':
                        lhd= dist_i(dfn=arg[0],dfd=arg[1],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
                    elif dist_i.name=='norm' or dist_i.name=='uniform' or dist_i.name=='expon':
                        lhd= dist_i(loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
                x = []
                for i in range(len(lhd)):
                    if weather_input=='temp_air':
                        if lhd[i][0]<-20:
                            lhd[i][0]=-20
                        elif lhd[i][0]>45:
                            lhd[i][0]=-20
                    else:
                        if lhd[i][0]<0:
                            lhd[i][0]=0
                        if lhd[i][0]>1000:
                            lhd[i][0]=1000
                    x.append(lhd[i][0])
                x.sort()
                for scenario in range(num_scenarios):
                    weather_scenario[weather_input][scenario].append(x[scenario+2])
                #print('here',loc, scale, weather_scenario[weather_input][i])
    iter = 0
    while iter<100: #Smoothing the radition and air temperatures
        for weather_input in weather_keys:
            for i in range(8760):
                for scenario in range(num_scenarios):
                    if weather_input=='temp_air':
                        if i<8759 and i>0:
                            if weather_scenario[weather_input][scenario][i]==45:
                                weather_scenario[weather_input][scenario][i] = (weather_scenario[weather_input][scenario][i-1]+weather_scenario[weather_input][scenario][i+1])/2
                            if weather_scenario[weather_input][scenario][i]==-20:
                                weather_scenario[weather_input][scenario][i] = (weather_scenario[weather_input][scenario][i-1]+weather_scenario[weather_input][scenario][i+1])/2
                            if  abs(weather_scenario[weather_input][scenario][i] -  weather_scenario[weather_input][scenario][i-1])>7 and abs(weather_scenario[weather_input][scenario][i] -  weather_scenario[weather_input][scenario][i+1])>7:
                                weather_scenario[weather_input][scenario][i] = (weather_scenario[weather_input][scenario][i-1]+weather_scenario[weather_input][scenario][i+1])/2
                    else:
                        if i<8759 and i>0:
                            if weather_scenario[weather_input][scenario][i]==1000:
                                weather_scenario[weather_input][scenario][i] = (weather_scenario[weather_input][scenario][i-1]+weather_scenario[weather_input][scenario][i+1])/2
                            if weather_scenario[weather_input][scenario][i]==0:
                                if weather_scenario[weather_input][scenario][i+1]!=0 and  weather_scenario[weather_input][scenario][i-1]!=0:
                                    weather_scenario[weather_input][scenario][i] = (weather_scenario[weather_input][scenario][i-1]+weather_scenario[weather_input][scenario][i+1])/2

        iter = iter +1

    scenario_genrated = {}
    scenario_genrated_normalized = {}
    for i_temp in range(num_scenarios):
        for i_solar in range(num_scenarios):
                scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)] = {'Temperature':weather_scenario['temp_air'][i_temp],
                 'GHI':weather_scenario['ghi'][i_solar],
                 'DNI':weather_scenario['dni'][i_solar],
                 'DHI':weather_scenario['dhi'][i_solar],
                 'GTI':weather_scenario['gti'][i_solar]}
                df_scenario_generated=pd.DataFrame(scenario_genrated['T_'+str(i_temp)+'_S_'+str(i_solar)])
                df_scenario_generated.to_csv(os.path.join(save_path,'T_'+str(i_temp)+'_S_'+str(i_solar)+'.csv'), index=False)

scenario_generation_results(path_test,weather_keys)
