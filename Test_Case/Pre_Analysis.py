### Pre-Analysis for the Design and Operation Planning of District Energy system ###
import os
import sys
import pandas as pd
import csv
from pathlib import Path
import subprocess
import DES_weather_analysis
from DES_weather_analysis import download_windsolar_data, weather_file_PDFs, scenario_generation_MCMC_interdependent,scenario_analysis_MCMC_interdependent,create_epws,eppy_connect_Eplus,clustring_kmediod_operation


path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city_DES =str(editable_data['city'])
state = editable_data['State']
start_year = int(editable_data['starting_year'])
end_year = int(editable_data['ending_year'])
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    #download_windsolar_data.download_meta_data(city_DES)
    #weather_file_PDFs.weather_PDS_data(path_test)
    #scenario_generation_MCMC_interdependent.scenario_generation_results(path_test)
    #scenario_analysis_MCMC_interdependent.scenario_analysis(path_test)
    #create_epws.create_epw_files(path_test)
    #eppy_connect_Eplus.run_Eplus(path_test)
    clustring_kmediod_operation.kmedoid_clusters(path_test,'total')
    clustring_kmediod_operation.kmedoid_clusters(path_test,'seperate')
