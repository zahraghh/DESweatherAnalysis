### Performing Two Stage Stochastic Programming for the Design of District Energy system ###
import os
import sys
import pandas as pd
import csv
from pathlib import Path
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
from pyomo.opt import SolverFactory
import DES_weather_analysis
from DES_weather_analysis import NSGA2_design_parallel_discrete_thermal
# I use platypus library to solve the muli-objective optimization problem:
# https://platypus.readthedocs.io/en/latest/getting-started.html
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values_design.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
results_folder = os.path.join(path_test,'Design_results')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if __name__ == "__main__":
    weather_data = []
    city_DES =str(editable_data['city'])
    state = editable_data['State']
    start_year = int(editable_data['starting_year'])
    end_year = int(editable_data['ending_year'])
    #end_year = 2011
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    num_scenarios = int(editable_data['num_scenarios'])
    for year in reversed(range(start_year,end_year+1)):
        weather_data.append(city_DES+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year))
    epw_names = []
    for i_temp in range(num_scenarios):
        for i_solar in range(num_scenarios):
            epw_names.append('T_'+str(i_temp)+'_S_'+str(i_solar))
    dict_EPWs= {}
    list_tmys= []
    list_fmys = []
    for i in range(5):
        if 'TMY'+str(i+1)+'_name' in editable_data.keys():
            TMY_name = editable_data['TMY'+str(i+1)+'_name']
            list_tmys.append(TMY_name)
        if 'FMY'+str(i+1)+'_name'  in editable_data.keys():
            FMY_name = editable_data['FMY'+str(i+1)+'_name']
            list_fmys.append(FMY_name)
    dict_EPWs['AMYs']=weather_data
    dict_EPWs['TMYs']=list_tmys
    dict_EPWs['FMYs']=list_fmys
    cluster_numbers= int(editable_data['Cluster numbers'])+2

    if editable_data['Perform two stage optimization']=='yes':
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                problem= NSGA2_design_parallel_discrete_thermal.TwoStageOpt(path_test,output_prefix)
                with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
                    algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
                    algorithm.run(int(editable_data['num_iterations']))
                NSGA2_design_parallel_discrete_thermal.results_extraction(problem, algorithm,path_test,output_prefix)
        for scenario in range(len(epw_names)):
            output_prefix =   'total_'+epw_names[scenario]+'_'
            problem= NSGA2_design_parallel_discrete_thermal.TwoStageOpt(path_test,output_prefix)
            with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
                algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
                algorithm.run(int(editable_data['num_iterations']))
            NSGA2_design_parallel_discrete_thermal.results_extraction(problem, algorithm,path_test,output_prefix)
