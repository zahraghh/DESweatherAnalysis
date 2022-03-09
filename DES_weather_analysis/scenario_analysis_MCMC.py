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
                for j in range(50):
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
