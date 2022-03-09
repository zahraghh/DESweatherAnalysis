import os
import pandas as pd
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import statistics
import itertools

JtokWh = 2.7778e-7
weight_factor =  [1.50558832,0.35786005,1.0]
path_test =  os.path.join(sys.path[0])
representative_days_path= os.path.join(path_test,'ScenarioReduction')
sizing_path = os.path.join(path_test, 'Design_results')
operation_path = os.path.join(path_test, 'Operation_results')

editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
editable_data_sizing_path =os.path.join(path_test, 'editable_values_design.csv')
editable_data_sizing = pd.read_csv(editable_data_sizing_path, header=None, index_col=0, squeeze=True).to_dict()[1]
num_scenarios = int(editable_data['num_scenarios'])
num_clusters= int(editable_data['Cluster numbers'])+2
population_size = int(editable_data['population_size'])
population_size_sizing = int(editable_data_sizing['population_size'])

idf_names = ['ASHRAE901_OfficeMedium_STD2019_Denver','ASHRAE901_Hospital_STD2019_Denver','ASHRAE901_RetailStandalone_STD2019_Denver']
thermal_eff_dict = {idf_names[0]:0.8,idf_names[1]:0.8125,idf_names[2]:0.82}

city=editable_data['city']
lat = float(editable_data['Latitude'])
lon = float(editable_data['Longitude'])
start_year = int(editable_data['starting_year'])
end_year = int(editable_data['ending_year'])
epw_names = []
for i_temp in range(num_scenarios):
    for i_solar in range(num_scenarios):
        epw_names.append('T_'+str(i_temp)+'_S_'+str(i_solar))
demand_directory = os.path.join(path_test, 'IDFBuildingsFiles')
# epw  main files
weather_data = []
weather_data_names =[]
weather_data_bar_names =[]
for year in range(start_year,end_year+1):
    weather_data.append(city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year))
    weather_data_names.append('AMY '+str(year))
    weather_data_bar_names.append('AMY \n'+str(year))

dict_EPWs= {}
dict_EPWs['AMYs']=weather_data
dict_EPWs['TMYs']=['USA_UT_Salt.Lake.City.Intl.AP.725720_TMY','USA_UT_Salt.Lake.City.725720_TMY2','USA_UT_Salt.Lake.City.Intl.AP.725720_TMY3']
#dict_EPWs['TMYs']=['USA_UT_Salt.Lake.City.Intl.AP.725720_TMY3']
dict_EPWs['FMYs']=['USA_Salt Lake City Intl AP_HadCM3-A2-'+str(2050),'USA_Salt Lake City Intl AP_HadCM3-A2-'+str(2080)]

dict_EPWs_names= {}
dict_EPWs_names['AMYs']=weather_data_names
dict_EPWs_names['TMYs']=['TMY','TMY2','TMY3']
#dict_EPWs_names['TMYs']=['TMY3']
dict_EPWs_names['FMYs']=['FMY '+str(2050),'FMY '+str(2080)]

dict_EPWs_bar_names= {}
dict_EPWs_bar_names['AMYs']=weather_data_bar_names
dict_EPWs_bar_names['TMYs']=['TMY \n','TMY2 \n','TMY3 \n']
dict_EPWs_bar_names['FMYs']=['FMY \n'+str(2050),'FMY \n'+str(2080)]

main_weather_epw = {}
output_directory = os.path.join(path_test, 'IDFBuildingsFiles')
results_compare = os.path.join(path_test, 'Results')
if not os.path.exists(results_compare):
    os.makedirs(results_compare)
years=list(range(1998,2020))
years= ['AMY \n'+str(i) for i in years]
years.append('TMY')
years.append('TMY2')
years.append('TMY3')
years.append('FMY \n'+str(2050))
years.append('FMY \n'+str(2080))

### Representative Days ###
def representative_day_function():
    global representative_days,weight_representative_day_main,weight_representative_day_scenario
    representative_days = defaultdict(list)
    weight_representative_day_scenario = defaultdict(list)
    weight_representative_day_main = defaultdict(list)
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            for representative_day in range(num_clusters):
                rep_day= pd.read_csv(os.path.join(representative_days_path,output_prefix + 'Represent_days_modified_'+str(representative_day)+ '.csv'))
                representative_days[output_prefix].append(rep_day)
                weight_representative_day_main[output_prefix].append(rep_day['Percent %']/100*365)
    for scenario in range(len(epw_names)):
        output_prefix = 'total_'+epw_names[scenario]+'_'
        for representative_day in range(num_clusters):
            rep_day= pd.read_csv(os.path.join(representative_days_path,output_prefix + 'Represent_days_modified_'+str(representative_day)+ '.csv'))
            representative_days[output_prefix].append(rep_day)
            weight_representative_day_scenario[output_prefix].append(rep_day['Percent %']/100*365)

### Energy Demands ###
def energy_demands():
    global elect_buildings,gas_buildings,cool_buildings,elect_annual,gas_annual,cool_annual,total_elect_buildings,total_gas_buildings,total_cool_buildings,total_elect_annual,total_gas_annual,total_cool_annual
    elect_buildings = defaultdict(list)
    gas_buildings = defaultdict(list)
    cool_buildings = defaultdict(list)
    elect_annual= defaultdict(list)
    gas_annual = defaultdict(list)
    cool_annual = defaultdict(list)
    total_elect_buildings= []
    total_gas_buildings = []
    total_cool_buildings = []
    total_elect_annual= []
    total_gas_annual = []
    total_cool_annual = []
    for scenario in range(len(epw_names)):
        sum_electricity_buildings = []
        sum_heating_buildings = []
        sum_cooling_buildings = []
        sum_electricity_annual = []
        sum_heating_annual = []
        sum_cooling_annual = []
        for building_type in idf_names:
            output_prefix =  building_type+'_'+epw_names[scenario]+'_mtr.csv'
            demand_data_path = os.path.join(demand_directory, output_prefix)
            data = pd.read_csv(demand_data_path)
            elect_data = (data['Electricity:Facility [J](Hourly)']-data['Heating:Electricity [J](Hourly)'] - data['Cooling:Electricity [J](Hourly)'])*JtokWh
            heat_data = (data['Gas:Facility [J](Hourly)']*thermal_eff_dict[building_type]+data['Heating:Electricity [J](Hourly)'])*JtokWh
            cool_data = (data['Cooling:Electricity [J](Hourly)'])*JtokWh
            elect_buildings[building_type].append(elect_data)
            gas_buildings[building_type].append(heat_data)
            cool_buildings[building_type].append(cool_data)
            elect_annual[building_type].append(sum(elect_data))
            gas_annual[building_type].append(sum(heat_data))
            cool_annual[building_type].append(sum(cool_data))
            sum_electricity_buildings.append(elect_data*weight_factor[idf_names.index(building_type)])
            sum_heating_buildings.append(heat_data*weight_factor[idf_names.index(building_type)])
            sum_cooling_buildings.append(cool_data*weight_factor[idf_names.index(building_type)])
            sum_electricity_annual.append(sum(elect_data*weight_factor[idf_names.index(building_type)]))
            sum_heating_annual.append(sum(heat_data*weight_factor[idf_names.index(building_type)]))
            sum_cooling_annual.append(sum(cool_data*weight_factor[idf_names.index(building_type)]))
        total_elect_buildings.append(sum(sum_electricity_buildings))
        total_gas_buildings.append(sum(sum_heating_buildings))
        total_cool_buildings.append(sum(sum_cooling_buildings))
        total_elect_annual.append(sum(sum_electricity_annual))
        total_gas_annual.append(sum(sum_heating_annual))
        total_cool_annual.append(sum(sum_cooling_annual))


    global elect_buildings_main,gas_buildings_main,cool_buildings_main,elect_annual_main,gas_annual_main,cool_annual_main,total_elect_buildings_main,total_gas_buildings_main,total_cool_buildings_main,total_elect_annual_main,total_gas_annual_main,total_cool_annual_main
    elect_buildings_main = defaultdict(list)
    gas_buildings_main = defaultdict(list)
    cool_buildings_main = defaultdict(list)
    elect_annual_main = defaultdict(list)
    gas_annual_main = defaultdict(list)
    cool_annual_main = defaultdict(list)
    total_elect_annual_main = []
    total_gas_annual_main = []
    total_cool_annual_main = []
    total_elect_buildings_main = []
    total_gas_buildings_main = []
    total_cool_buildings_main = []
    global output_prefix_short
    output_prefix_short ={}
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            output_prefix_short[output_prefix] =  dict_EPWs_names[key][dict_EPWs[key].index(epw_file_name)]
            sum_electricity_buildings_main = []
            sum_heating_buildings_main = []
            sum_cooling_buildings_main = []
            sum_electricity_annual_main = []
            sum_heating_annual_main = []
            sum_cooling_annual_main = []
            for building_type in idf_names:
                output_prefix =  building_type+'_'+epw_file_name+'_mtr.csv'
                demand_data_path = os.path.join(demand_directory, output_prefix)
                data = pd.read_csv(demand_data_path)
                elect_data = (data['Electricity:Facility [J](Hourly)']-data['Heating:Electricity [J](Hourly)'] - data['Cooling:Electricity [J](Hourly)'])*JtokWh
                heat_data = (data['Gas:Facility [J](Hourly)']*thermal_eff_dict[building_type]+data['Heating:Electricity [J](Hourly)'])*JtokWh
                cool_data = (data['Cooling:Electricity [J](Hourly)'])*JtokWh
                elect_buildings_main[building_type].append(elect_data)
                gas_buildings_main[building_type].append(heat_data)
                cool_buildings_main[building_type].append(cool_data)
                elect_annual_main[building_type].append(sum(elect_data))
                gas_annual_main[building_type].append(sum(heat_data))
                cool_annual_main[building_type].append(sum(cool_data))
                sum_electricity_buildings_main.append(elect_data*weight_factor[idf_names.index(building_type)])
                sum_heating_buildings_main.append(heat_data*weight_factor[idf_names.index(building_type)])
                sum_cooling_buildings_main.append(cool_data*weight_factor[idf_names.index(building_type)])
                sum_electricity_annual_main.append(sum(elect_data*weight_factor[idf_names.index(building_type)]))
                sum_heating_annual_main.append(sum(heat_data*weight_factor[idf_names.index(building_type)]))
                sum_cooling_annual_main.append(sum(cool_data*weight_factor[idf_names.index(building_type)]))
            total_elect_buildings_main.append(sum(sum_electricity_buildings_main))
            total_gas_buildings_main.append(sum(sum_heating_buildings_main))
            total_cool_buildings_main.append(sum(sum_cooling_buildings_main))
            total_elect_annual_main.append(sum(sum_electricity_annual_main))
            total_gas_annual_main.append(sum(sum_heating_annual_main))
            total_cool_annual_main.append(sum(sum_cooling_annual_main))

    j = 0
def generate_combo_plots(mode):
    SMALL_SIZE = 30
    MEDIUM_SIZE = 32
    BIGGER_SIZE = 38
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
    plt.rcParams["figure.figsize"] = (30,20)
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                 for i in range(27)]
    if mode=='seperate':
        marker = itertools.cycle(('v','+','s','^','o','x','*'))
        for building_type in idf_names:
            plt.figure()
            j=0
            for key in dict_EPWs.keys():
                for epw_file_name in dict_EPWs[key]:
                    output_prefix =  'total_'+epw_file_name+'_'
                    label = output_prefix_short[output_prefix]

                    if key=='AMYs':
                        year_selected_number=int(label.replace('AMY',''))
                        if year_selected_number==2019 or year_selected_number==2018 or year_selected_number==2016 or year_selected_number==2014 or year_selected_number==2012:
                            plt.scatter(cool_annual_main[building_type][j]/1000,gas_annual_main[building_type][j]/1000,s=400, cmap=cmap, label = label,marker=next(marker))
                            plt.annotate(label,xy=(cool_annual_main[building_type][j]/1000,gas_annual_main[building_type][j]/1000),xytext=(cool_annual_main[building_type][j]/1000*1.0005,gas_annual_main[building_type][j]/1000*1.0005),
                                        arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
                            j = j+1
                    elif key=='TMYs':
                        if label=='TMY3':
                            plt.scatter(cool_annual_main[building_type][j]/1000,gas_annual_main[building_type][j]/1000,s=400, cmap=cmap, label = label,marker=next(marker))
                            plt.annotate(label,xy=(cool_annual_main[building_type][j]/1000,gas_annual_main[building_type][j]/1000),xytext=(cool_annual_main[building_type][j]/1000*1.0005,gas_annual_main[building_type][j]/1000*1.0005),
                                        arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
                            j = j+1

                    elif key=='FMYs':
                        plt.scatter(cool_annual_main[building_type][j]/1000,gas_annual_main[building_type][j]/1000,s=400, cmap=cmap, label = label,marker=next(marker))
                        plt.annotate(label,xy=(cool_annual_main[building_type][j]/1000,gas_annual_main[building_type][j]/1000),xytext=(cool_annual_main[building_type][j]/1000*1.0005,gas_annual_main[building_type][j]/1000*1.0005),
                                    arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
                        j = j+1

            plt.xlabel('Total Cooling Demand (MWh)')
            plt.ylabel('Total Hot Water Demand (MWh)')
            plt.savefig(os.path.join(results_compare,building_type+'_annual_main_combo_demands_WC'+'.png'),dpi=100,facecolor='w',bbox_inches='tight')
            plt.close()
        marker_list = ['v','+','s','^','o','x','*','s','>','<']
        color_list=  [ 'tab:blue', 'tab:orange','tab:green','black','yellow','tab:red','tab:cyan','tab:olive','peru','tab:purple']
        for building_type in idf_names:
            plt.figure()
            label_dict = {}
            for scenario in range(len(epw_names)):
                key_label = round(cool_annual[building_type][scenario]/1000,0)
                #if key_label not in label_dict.keys():
                    #label_dict[key_label] = epw_names[scenario]
                label_short= epw_names[scenario].replace('_','')
                marker = marker_list[int(label_short[1])]
                color = color_list[int(label_short[3])]

                if (int(label_short[1])==0 or int(label_short[1])==5 or int(label_short[1])==9) and (int(label_short[3])==0 or int(label_short[3])==5 or int(label_short[3])==9):
                    if int(label_short[1])==0:
                        label_T='Tmin'
                    elif int(label_short[1])==5:
                        label_T='Tmed'
                    elif int(label_short[1])==9:
                        label_T='Tmax'
                    if int(label_short[3])==0:
                        label_S='Smin'
                    elif int(label_short[3])==5:
                        label_S='Smed'
                    elif int(label_short[3])==9:
                        label_S='Smax'
                    label =  label_T + label_S
                    if building_type==idf_names[1]:
                        weight_factor_pareto_front =0.9955
                    else:
                        weight_factor_pareto_front = 1
                    plt.scatter(cool_annual[building_type][scenario]/1000,gas_annual[building_type][scenario]/1000,color=color,marker=marker,s=300, cmap=cmap, label=label_short)
                    plt.annotate(label,xy=(cool_annual[building_type][scenario]/1000,gas_annual[building_type][scenario]/1000),xytext=(cool_annual[building_type][scenario]/1000*1.005*weight_factor_pareto_front,gas_annual[building_type][scenario]/1000),
                                arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
            plt.xlabel('Total Cooling Demand (MWh)')
            plt.ylabel('Total Hot Water Demand (MWh)')
            plt.savefig(os.path.join(results_compare,building_type+'_annual_scenario_combo_demands_WC'+'.png'),dpi=100,facecolor='w',bbox_inches='tight')
            plt.close()

    elif mode =='total':
        marker = itertools.cycle(('v','+','s','^','o','x','*'))
        plt.figure()
        j=0
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                label = output_prefix_short[output_prefix]
                if key=='AMYs':
                    year_selected_number=int(label.replace('AMY',''))
                    if year_selected_number==2019 or year_selected_number==2018 or year_selected_number==2016 or year_selected_number==2014 or year_selected_number==2012:
                        plt.scatter(total_cool_annual_main[j]/1000,total_gas_annual_main[j]/1000,s=400, cmap=cmap, label = label,marker=next(marker))
                        plt.annotate(label,xy=(total_cool_annual_main[j]/1000,total_gas_annual_main[j]/1000),xytext=(total_cool_annual_main[j]/1000*1.0005,total_gas_annual_main[j]/1000*1.0005),
                                    arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
                        j = j+1
                elif key=='TMYs':
                    if label=='TMY3':
                        plt.scatter(total_cool_annual_main[j]/1000,total_gas_annual_main[j]/1000,s=400, cmap=cmap, label = label,marker=next(marker))
                        plt.annotate(label,xy=(total_cool_annual_main[j]/1000,total_gas_annual_main[j]/1000),xytext=(total_cool_annual_main[j]/1000*1.0005,total_gas_annual_main[j]/1000*1.0005),
                                    arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
                        j = j+1

                elif key=='FMYs':
                    plt.scatter(total_cool_annual_main[j]/1000,total_gas_annual_main[j]/1000,s=400, cmap=cmap, label = label,marker=next(marker))
                    plt.annotate(label,xy=(total_cool_annual_main[j]/1000,total_gas_annual_main[j]/1000),xytext=(total_cool_annual_main[j]/1000*1.0005,total_gas_annual_main[j]/1000*1.0005),
                                arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
                    j = j+1

        plt.xlabel('Total Cooling Demand (MWh)')
        plt.ylabel('Total Hot Water Demand (MWh)')
        plt.savefig(os.path.join(results_compare,'total_annual_main_combo_demands_WC'+'.png'),dpi=100,facecolor='w',bbox_inches='tight')
        plt.close()

        marker_list = ['v','+','s','^','o','x','*','s','>','<']
        color_list=  [ 'tab:blue', 'tab:orange','tab:green','black','yellow','tab:red','tab:cyan','tab:olive','peru','tab:purple']
        label_dict = {}
        for scenario in range(len(epw_names)):
            key_label = round(total_cool_annual[scenario]/1000,0)
            #if key_label not in label_dict.keys():
            #    label_dict[key_label] = epw_names[scenario]
            label_short= epw_names[scenario].replace('_','')
            marker = marker_list[int(label_short[1])]
            color = color_list[int(label_short[3])]
            if (int(label_short[1])==0 or int(label_short[1])==5 or int(label_short[1])==9) and (int(label_short[3])==0 or int(label_short[3])==5 or int(label_short[3])==9):
                if int(label_short[1])==0:
                    label_T='Tmin'
                elif int(label_short[1])==5:
                    label_T='Tmed'
                elif int(label_short[1])==9:
                    label_T='Tmax'
                if int(label_short[3])==0:
                    label_S='Smin'
                elif int(label_short[3])==5:
                    label_S='Smed'
                elif int(label_short[3])==9:
                    label_S='Smax'
                label =  label_T + label_S
                plt.scatter(total_cool_annual[scenario]/1000,total_gas_annual[scenario]/1000,s=300,c=color,marker=marker, cmap=cmap, label=label_short)
                plt.annotate(label,xy=(total_cool_annual[scenario]/1000,total_gas_annual[scenario]/1000),xytext=(total_cool_annual[scenario]/1000*1.001,total_gas_annual[scenario]/1000*1.001),
                            arrowprops=dict(arrowstyle="-"),fontsize=MEDIUM_SIZE)
        plt.xlabel('Total Cooling Demand (MWh)')
        plt.ylabel('Total Hot Water Demand (MWh)')
        plt.savefig(os.path.join(results_compare,'total_annual_scenario_combo_demands_WC'+'.png'),dpi=100,facecolor='w',bbox_inches='tight')
        plt.close()
def stats_energy_demands():
    cols_revised = ['Office Medium','Hospital','Retail stand-alone', 'Total']
    weight_factor_dict = {idf_names[0]:weight_factor[0],idf_names[1]:weight_factor[1],idf_names[2]:weight_factor[2]}
    stats_table_seperate = defaultdict(list)
    k=0
    for building_type in idf_names:
        #stats_table_seperate[k].append(round(np.mean(elect_annual_main[building_type])*weight_factor_dict[building_type]/1000,2))
        #stats_table_seperate[k].append(round(np.std(elect_annual_main[building_type])*weight_factor_dict[building_type]/1000,2))
        #stats_table_seperate[k].append(round(np.std(elect_annual_main[building_type])*100/np.mean(elect_annual_main[building_type]),2))
        stats_table_seperate[k].append(round(np.mean(gas_annual_main[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(gas_annual_main[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(gas_annual_main[building_type])*100/np.mean(gas_annual_main[building_type]),2))
        stats_table_seperate[k].append(round(np.mean(cool_annual_main[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(cool_annual_main[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(cool_annual_main[building_type])*100/np.mean(cool_annual_main[building_type]),2))
        k = k+1
    stats_table_total = []
    #stats_table_total.append(round(np.mean(total_elect_annual_main)/1000,2))
    #stats_table_total.append(round(np.std(total_elect_annual_main)/1000,2))
    #stats_table_total.append(round(np.std(total_elect_annual_main)*100/np.mean(total_elect_annual_main),2))
    stats_table_total.append(round(np.mean(total_gas_annual_main)/1000,2))
    stats_table_total.append(round(np.std(total_gas_annual_main)/1000,2))
    stats_table_total.append(round(np.std(total_gas_annual_main)*100/np.mean(total_gas_annual_main),2))
    stats_table_total.append(round(np.mean(total_cool_annual_main)/1000,2))
    stats_table_total.append(round(np.std(total_cool_annual_main)/1000,2))
    stats_table_total.append(round(np.std(total_cool_annual_main)*100/np.mean(total_cool_annual_main),2))
    statistics_table = {#'Elect Mean': [stats_table_seperate[0][0],stats_table_seperate[1][0],stats_table_seperate[2][0],stats_table_total[0]],
    #'Elect STD': [stats_table_seperate[0][1],stats_table_seperate[1][1],stats_table_seperate[2][1],stats_table_total[1]],
    #'CV \% Elect': [stats_table_seperate[0][2],stats_table_seperate[1][2],stats_table_seperate[2][2],stats_table_total[2]],
    'Heat Mean': [stats_table_seperate[0][3],stats_table_seperate[1][3],stats_table_seperate[2][3],stats_table_total[3]],
    'Heat STD': [stats_table_seperate[0][4],stats_table_seperate[1][4],stats_table_seperate[2][4],stats_table_total[4]],
    'CV \% Heat': [stats_table_seperate[0][5],stats_table_seperate[1][5],stats_table_seperate[2][5],stats_table_total[5]],
    'Cool Mean': [stats_table_seperate[0][6],stats_table_seperate[1][6],stats_table_seperate[2][6],stats_table_total[6]],
    'Cool STD': [stats_table_seperate[0][7],stats_table_seperate[1][7],stats_table_seperate[2][7],stats_table_total[7]],
    'CV \% Cool': [stats_table_seperate[0][8],stats_table_seperate[1][8],stats_table_seperate[2][8],stats_table_total[8]]}
    df_statistics_table= pd.DataFrame(statistics_table)
    df_statistics_table.insert(0, "", cols_revised, True)
    for i in range(1,len(df_statistics_table.columns)*2):
        if i%2!=0:
            df_statistics_table.insert(i, "&", ["&"]*len(df_statistics_table), True)

    df_statistics_table.insert(i, "\\\\ \hline", ["\\\\ \hline"]*len(df_statistics_table), True)
    df_statistics_table.to_csv(os.path.join(results_compare,'stats_main_seperate_demand_WC_table.csv'))

    stats_table_seperate = defaultdict(list)
    weight_factor_dict = {idf_names[0]:weight_factor[0],idf_names[1]:weight_factor[1],idf_names[2]:weight_factor[2]}

    k=0
    for building_type in idf_names:
        #print(
        #building_type,np.std(elect_annual[building_type])*weight_factor_dict[building_type]/1000)
        #stats_table_seperate[k].append(round(np.mean(elect_annual[building_type])*weight_factor_dict[building_type]/1000,2))
        #stats_table_seperate[k].append(round(np.std(elect_annual[building_type])*weight_factor_dict[building_type]/1000,2))
        #stats_table_seperate[k].append(round(np.std(elect_annual[building_type])*100/np.mean(elect_annual[building_type]),2))
        stats_table_seperate[k].append(round(np.mean(gas_annual[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(gas_annual[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(gas_annual[building_type])*100/np.mean(gas_annual[building_type]),2))
        stats_table_seperate[k].append(round(np.mean(cool_annual[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(cool_annual[building_type])*weight_factor_dict[building_type]/1000,2))
        stats_table_seperate[k].append(round(np.std(cool_annual[building_type])*100/np.mean(cool_annual[building_type]),2))
        k = k+1
    stats_table_total = []
    #stats_table_total.append(round(np.mean(total_elect_annual)/1000,2))
    #stats_table_total.append(round(np.std(total_elect_annual)/1000,2))
    #stats_table_total.append(round(np.std(total_elect_annual)*100/np.mean(total_elect_annual),2))
    stats_table_total.append(round(np.mean(total_gas_annual)/1000,2))
    stats_table_total.append(round(np.std(total_gas_annual)/1000,2))
    stats_table_total.append(round(np.std(total_gas_annual)*100/np.mean(total_gas_annual),2))
    stats_table_total.append(round(np.mean(total_cool_annual)/1000,2))
    stats_table_total.append(round(np.std(total_cool_annual)/1000,2))
    stats_table_total.append(round(np.std(total_cool_annual)*100/np.mean(total_cool_annual),2))
    statistics_table = {#'Elect Mean': [stats_table_seperate[0][0],stats_table_seperate[1][0],stats_table_seperate[2][0],stats_table_total[0]],
    #'Elect STD': [stats_table_seperate[0][1],stats_table_seperate[1][1],stats_table_seperate[2][1],stats_table_total[1]],
    #'CV \% Elect': [stats_table_seperate[0][2],stats_table_seperate[1][2],stats_table_seperate[2][2],stats_table_total[2]],
    'Heat Mean': [stats_table_seperate[0][3],stats_table_seperate[1][3],stats_table_seperate[2][3],stats_table_total[3]],
    'Heat STD': [stats_table_seperate[0][4],stats_table_seperate[1][4],stats_table_seperate[2][4],stats_table_total[4]],
    'CV \% Heat': [stats_table_seperate[0][5],stats_table_seperate[1][5],stats_table_seperate[2][5],stats_table_total[5]],
    'Cool Mean': [stats_table_seperate[0][6],stats_table_seperate[1][6],stats_table_seperate[2][6],stats_table_total[6]],
    'Cool STD': [stats_table_seperate[0][7],stats_table_seperate[1][7],stats_table_seperate[2][7],stats_table_total[7]],
    'CV \% Cool': [stats_table_seperate[0][8],stats_table_seperate[1][8],stats_table_seperate[2][8],stats_table_total[8]]}
    df_statistics_table= pd.DataFrame(statistics_table)
    df_statistics_table.insert(0, "", cols_revised, True)
    for i in range(1,len(df_statistics_table.columns)*2):
        if i%2!=0:
            df_statistics_table.insert(i, "&", ["&"]*len(df_statistics_table), True)

    df_statistics_table.insert(i, "\\\\ \hline", ["\\\\ \hline"]*len(df_statistics_table), True)
    df_statistics_table.to_csv(os.path.join(results_compare,'stats_scenario_seperate_WC_demand_table.csv'))
def bar_energy_demands(mode):
    SMALL_SIZE = 30
    MEDIUM_SIZE = 32
    BIGGER_SIZE = 38
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
    plt.rcParams["figure.figsize"] = (45,15)

    if mode=='seperate':
        for building_type in idf_names:
            print(building_type, 'highest elect',  years[elect_annual_main[building_type].index(np.max(elect_annual_main[building_type]))],
            ' heat',  years[gas_annual_main[building_type].index(np.max(gas_annual_main[building_type]))],
            ' cool',  years[cool_annual_main[building_type].index(np.max(cool_annual_main[building_type]))],
            )
            print(building_type, 'lowest elect',  years[elect_annual_main[building_type].index(np.min(elect_annual_main[building_type]))],
            ' heat',  years[gas_annual_main[building_type].index(np.min(gas_annual_main[building_type]))],
            ' cool',  years[cool_annual_main[building_type].index(np.min(cool_annual_main[building_type]))],
            )
            r = np.arange(n)
            width = 0.25
            plt.figure()
            #plt.bar(r,[number/1000 for number in elect_annual_main[building_type]],width = width,color='darkorange', edgecolor = 'black',label='Annual Electricity')
            plt.bar(r,[number/1000 for number in gas_annual_main[building_type]],width = width,color='darkred', edgecolor = 'black',label = 'Annual Hot Water')
            plt.bar(r+width,[number/1000 for number in cool_annual_main[building_type]],width = width,color='darkblue', edgecolor = 'black',label = 'Annual Cooling')

            plt.xlabel('Weather Files')
            plt.ylabel('Energy Demands (MWh)')
            plt.xticks(r + width/2,years)

            #plt.yticks
            plt.legend(loc='center left')
            plt.ticklabel_format(style='plain', axis='y')
            #plt.title('annual energy demands of' + building_type)
            plt.savefig(os.path.join(results_compare,building_type+'_bar_annual_main_demands_WC'+'.png'),dpi=100,facecolor='w',bbox_inches='tight')
            plt.close()
    elif mode =='total':
            print(#'total', 'highest elect',  years[total_elect_annual_main.index(np.max(total_elect_annual_main))],
            ' heat',  years[total_gas_annual_main.index(np.max(total_gas_annual_main))],
            ' cool',  years[total_cool_annual_main.index(np.max(total_cool_annual_main))],
            )
            print(#'total', 'lowest elect',  years[total_elect_annual_main.index(np.min(total_elect_annual_main))],
            ' heat',  years[total_gas_annual_main.index(np.min(total_gas_annual_main))],
            ' cool',  years[total_cool_annual_main.index(np.min(total_cool_annual_main))],
            )
            print(#'total range','elect', (np.max(total_elect_annual_main)-np.min(total_elect_annual_main))/1000,
            'heat',(np.max(total_gas_annual_main)-np.min(total_gas_annual_main))/1000,
            'cool', (np.max(total_cool_annual_main)-np.min(total_cool_annual_main))/1000)

            n=len(total_elect_annual_main)
            r = np.arange(n)
            width = 0.25
            plt.figure()
            #plt.bar(r,[number/1000 for number in total_elect_annual_main],width = width,color='darkorange', edgecolor = 'black',label='Annual Electricity')
            plt.bar(r,[number/1000 for number in total_gas_annual_main],width = width,color='darkred', edgecolor = 'black',label = 'Annual Hot Water')
            plt.bar(r+width,[number/1000 for number in total_cool_annual_main],width = width,color='darkblue', edgecolor = 'black',label = 'Annual Cooling')

            plt.xlabel('Weather Files')
            plt.ylabel('Energy Demands (MWh)')
            plt.xticks(r + width/2,years)
            #plt.yticks(fontsize=BIGGER_SIZE)
            plt.legend(loc='center left')
            plt.ticklabel_format(style='plain', axis='y')
            #plt.title('Total annual energy demands')
            plt.savefig(os.path.join(results_compare,'total_annual_main_demands_WC'+'.png'),dpi=100,facecolor='w',bbox_inches='tight')
            plt.close()
def hist_scenarios(mode):
    SMALL_SIZE = 20
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28
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
    plt.rcParams["figure.figsize"] = (10,8)
    if mode=='seperate':
        for building_type in idf_names:
            #plt.ylabel('Percentage %')
            #plt.hist([number/1000 for number in elect_annual[building_type]],color='darkorange',bins=10,weights=np.ones(len([number/1000 for number in elect_annual[building_type]]))*100 / len([number/1000 for number in elect_annual[building_type]]))
            #plt.xlabel('Total Electricity Demand (MWh)')
            #plt.savefig(os.path.join(results_compare,'hist_'+building_type+'_annual_main_electricity_WC_demand'+'.png'),dpi=100,facecolor='w')
            plt.close()
            plt.ylabel('Percentage %')
            plt.hist([number/1000 for number in gas_annual[building_type]],color='darkred',bins=10,weights=np.ones(len([number/1000 for number in gas_annual[building_type]]))*100 / len([number/1000 for number in gas_annual[building_type]]))
            plt.xlabel('Total Heating Demand (MWh)')
            plt.savefig(os.path.join(results_compare,'hist_'+building_type+'_annual_main_heating_WC_demand'+'.png'),dpi=100,facecolor='w')
            plt.close()
            plt.ylabel('Percentage %')
            plt.hist([number/1000 for number in cool_annual[building_type]],color='darkblue',bins=10,weights=np.ones(len([number/1000 for number in cool_annual[building_type]]))*100 / len([number/1000 for number in cool_annual[building_type]]))
            plt.xlabel('Total Cooling Demand (MWh)')
            plt.savefig(os.path.join(results_compare,'hist_'+building_type+'_annual_main_cooling_WC_demand'+'.png'),dpi=100,facecolor='w')
            plt.close()
    elif mode =='total':
        #plt.ylabel('Percentage %')
        #plt.hist([number/1000 for number in total_elect_annual],color='darkorange',bins=10,weights=np.ones(len([number/1000 for number in total_elect_annual]))*100 / len([number/1000 for number in total_elect_annual]))
        #plt.xlabel('Total Electricity Demand (MWh)')
        #plt.savefig(os.path.join(results_compare,'hist_total_annual_main_electricity_WC_demand'+'.png'),dpi=100,facecolor='w')
        plt.close()
        plt.hist([number/1000 for number in total_gas_annual],color='darkred',bins=10,weights=np.ones(len([number/1000 for number in total_gas_annual]))*100 / len([number/1000 for number in total_gas_annual]))
        plt.ylabel('Percentage %')
        plt.xlabel('Total Heating Demand (MWh)')
        plt.savefig(os.path.join(results_compare,'hist_total_annual_main_heating_WC_demand'+'.png'),dpi=100,facecolor='w')
        plt.close()
        plt.hist([number/1000 for number in total_gas_annual],color='darkblue',bins=10,weights=np.ones(len([number/1000 for number in total_gas_annual]))*100 / len([number/1000 for number in total_gas_annual]))
        plt.ylabel('Percentage %')
        plt.xlabel('Total Cooling Demand (MWh)')
        plt.savefig(os.path.join(results_compare,'hist_total_annual_main_cooling_WC_demand'+'.png'),dpi=100,facecolor='w')
        plt.close()


energy_demands()
generate_combo_plots('seperate')
generate_combo_plots('total')
#bar_energy_demands('seperate')
#bar_energy_demands('total')
#hist_scenarios('total')
#hist_scenarios('seperate')
#stats_energy_demands()

### Sizing of DES ###
def sizing():
    global annual_df_object_sizing_main,annual_df_operation_sizing_main
    annual_df_object_sizing_main= {}
    annual_df_operation_sizing_main = {}
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            file_name = output_prefix+city+'_Discrete_EF_'+str(float(editable_data_sizing['renewable percentage']) )+'_design_'+str(int(editable_data_sizing['num_iterations']))+'_'+str(editable_data_sizing['population_size'])+'_'+str(editable_data_sizing['num_processors'])+'_processors'
            annual_df_object_sizing_main[output_prefix]=pd.read_csv(os.path.join(sizing_path,file_name, 'objectives.csv'))
            annual_df_operation_sizing_main[output_prefix]=pd.read_csv(os.path.join(sizing_path,file_name, 'sizing_all.csv'))
    global  annual_df_object_sizing_scenario, annual_df_operation_sizing_scenario
    annual_df_object_sizing_scenario= {}
    annual_df_operation_sizing_scenario = {}
    for scenario in range(len(epw_names)):
        output_prefix =   'total_'+epw_names[scenario]+'_'
        file_name = output_prefix+city+'_Discrete_EF_'+str(float(editable_data_sizing['renewable percentage']) )+'_design_'+str(int(editable_data_sizing['num_iterations']))+'_'+str(editable_data_sizing['population_size'])+'_'+str(editable_data_sizing['num_processors'])+'_processors'
        annual_df_object_sizing_scenario[output_prefix]=pd.read_csv(os.path.join(sizing_path,file_name , 'objectives.csv'))
        annual_df_operation_sizing_scenario[output_prefix]=pd.read_csv(os.path.join(sizing_path,file_name, 'sizing_all.csv'))
def main_paretofront_sizing():
    global sorted_annual_df_object_sizing_main,output_prefix_short
    SMALL_SIZE = 22
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28
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
    plt.rcParams["figure.figsize"] = (30,15)
    plt.figure()
    j=0
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                 for i in range(27)]
    sorted_cost = []
    output_prefix_short ={}
    marker = itertools.cycle(('v','+','s','^','o','x','*'))
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            output_prefix_short[output_prefix] =  dict_EPWs_names[key][dict_EPWs[key].index(epw_file_name)]
            annual_df_object_sizing_main[output_prefix]=annual_df_object_sizing_main[output_prefix].sort_values('Cost ($)')
            annual_df_object_sizing_main[output_prefix]=annual_df_object_sizing_main[output_prefix].reset_index()
            if key is 'AMYs':
                year_selected_number=int(output_prefix_short[output_prefix].replace('AMY',''))
                if year_selected_number==2019 or year_selected_number==2018 or year_selected_number==2016 or year_selected_number==2014 or year_selected_number==2012:
                    sorted_cost.append(annual_df_object_sizing_main[output_prefix]['Cost ($)'][0]/10**6)
            elif key is 'TMYs':
                if epw_file_name=='USA_UT_Salt.Lake.City.Intl.AP.725720_TMY3':
                    sorted_cost.append(annual_df_object_sizing_main[output_prefix]['Cost ($)'][0]/10**6)
            else:
                sorted_cost.append(annual_df_object_sizing_main[output_prefix]['Cost ($)'][0]/10**6)

    sorted_cost = sorted(sorted_cost)
    sorted_annual_df_object_sizing_main = {}
    for i in sorted_cost:
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                if annual_df_object_sizing_main[output_prefix]['Cost ($)'][0]/10**6 == i:
                    sorted_annual_df_object_sizing_main[output_prefix] =annual_df_object_sizing_main[output_prefix]
    sorted_cost_scenario = []
    for scenario in range(len(epw_names)):
        output_prefix =   'total_'+epw_names[scenario]+'_'
        annual_df_object_sizing_scenario[output_prefix]=annual_df_object_sizing_scenario[output_prefix].sort_values('Cost ($)')
        annual_df_object_sizing_scenario[output_prefix]=annual_df_object_sizing_scenario[output_prefix].reset_index()
        sorted_cost_scenario.append(annual_df_object_sizing_scenario[output_prefix]['Cost ($)'][0]/10**6)
    sorted_cost_scenario = sorted(sorted_cost_scenario)
    sorted_annual_df_object_sizing_scenario = {}
    for i in sorted_cost_scenario:
        for scenario in range(len(epw_names)):
            output_prefix =   'total_'+epw_names[scenario]+'_'
            if annual_df_object_sizing_scenario[output_prefix]['Cost ($)'][0]/10**6 == i:
                sorted_annual_df_object_sizing_scenario[output_prefix] =annual_df_object_sizing_scenario[output_prefix]
    j=0
    #fig, ax = plt.subplots()
    for key in sorted_annual_df_object_sizing_main:
        output_prefix =  key
        cost = [i/10**6 for i in sorted_annual_df_object_sizing_main[output_prefix]['Cost ($)']]
        emissions = [j/10**6 for j in sorted_annual_df_object_sizing_main[output_prefix]['Emission (kg CO2)']]
        label = output_prefix_short[output_prefix]
        #plt.scatter(cost,emissions,c=color[j], s=100, cmap=cmap,marker=next(marker))
        #plt.title('Cost and emissions trade-off')
        if j==0:
            plt.annotate(label,xy=(cost[-1], emissions[-1]),xytext=(cost[-1]*1.05, emissions[-1]),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'tab:blue'

        elif j==1:
            plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.9, emissions[0]*1.15),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'tab:orange'

        elif j==2:
            plt.annotate(label,xy=(cost[-1], emissions[-1]),xytext=(cost[-1]*1.05, emissions[-1]*0.8),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'tab:green'

        elif j==3:
            plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.85, emissions[0]*1.1),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'tab:purple'

        elif j==4:
            plt.annotate(label,xy=(cost[-1], emissions[-1]),xytext=(cost[-1]*1.05, emissions[-1]*0.8),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'black'

        elif j==5:
            plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.85, emissions[0]*1.1),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'tab:red'

        elif j==6:
            plt.annotate(label,xy=(cost[-1], emissions[-1]),xytext=(cost[-1]*0.9, emissions[-1]*1.4),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'tab:cyan'

        elif j==7:
            plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.85, emissions[0]*1.2),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
            color = 'tab:olive'
        plt.scatter(cost,emissions,c=color, s=100, cmap=cmap,marker=next(marker))

        j = j+1
    j=0
    #plt.legend()
    plt.xlabel("Cost (million $)")
    plt.ylabel("Emissions (million kg $CO_2$)")
    plt.savefig(os.path.join(results_compare ,'ParetoFront_sizing.png'),dpi=100,facecolor='w',bbox_inches='tight')
def scenario_paretofront_sizing():
    SMALL_SIZE = 22
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28
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
    plt.rcParams["figure.figsize"] = (30,15)
    plt.figure()
    j=0

    sorted_cost = []
    output_prefix_short ={}
    sorted_cost_scenario = []
    for scenario in range(len(epw_names)):
        output_prefix =   'total_'+epw_names[scenario]+'_'
        annual_df_object_sizing_scenario[output_prefix]=annual_df_object_sizing_scenario[output_prefix].sort_values('Cost ($)')
        annual_df_object_sizing_scenario[output_prefix]=annual_df_object_sizing_scenario[output_prefix].reset_index()
        sorted_cost_scenario.append(annual_df_object_sizing_scenario[output_prefix]['Cost ($)'][0]/10**6)
    sorted_cost_scenario = sorted(sorted_cost_scenario)
    sorted_annual_df_object_sizing_scenario = {}
    for i in sorted_cost_scenario:
        for scenario in range(len(epw_names)):
            output_prefix =   'total_'+epw_names[scenario]+'_'
            label = output_prefix.replace('_','').replace('total','')
            if annual_df_object_sizing_scenario[output_prefix]['Cost ($)'][0]/10**6 == i and (int(label[1])==0 or int(label[1])==5 or int(label[1])==9) and (int(label[3])==0 or int(label[3])==5 or int(label[3])==9):
                sorted_annual_df_object_sizing_scenario[output_prefix] =annual_df_object_sizing_scenario[output_prefix]

    j=0
    marker_list = ['v','+','s','^','o','x','*','s','>','<']
    color_list=  [ 'tab:blue', 'tab:orange','tab:green','black','yellow','tab:red','tab:cyan','tab:olive','peru','tab:purple']
    for key in sorted_annual_df_object_sizing_scenario:
        output_prefix =  key
        cost = [i/10**6 for i in sorted_annual_df_object_sizing_scenario[output_prefix]['Cost ($)']]
        emissions = [j/10**6 for j in sorted_annual_df_object_sizing_scenario[output_prefix]['Emission (kg CO2)']]
        label = key.replace('_','').replace('total','')
        int_marker= int(label[1])
        int_color = int(label[3])
        #print(int_marker, type(int_marker), marker[int_marker],len( marker))
        marker = marker_list[int_marker]
        color = color_list[int_color]
        plt.scatter(cost,emissions,c=color, s=300, cmap=cmap, label = label,marker=marker)
        #plt.title('Cost and emissions trade-off')
        plt.xlabel("Cost (million $)")
        plt.ylabel("Emissions (million kg $CO_2$)")
        if int(label[1])==0:
            label_T='Tmin'
        elif int(label[1])==5:
            label_T='Tmed'
        elif int(label[1])==9:
            label_T='Tmax'
        if int(label[3])==0:
            label_S='Smin'
        elif int(label[3])==5:
            label_S='Smed'
        elif int(label[3])==9:
            label_S='Smax'
        label =  label_T + label_S
        if j == 0:
            plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.90, emissions[0]*1.1),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
        else:
            plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.88, emissions[0]*1.1),
            arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)
        j=j+1
    #plt.legend()
    plt.savefig(os.path.join(results_compare ,'scenario_ParetoFront_sizing.png'),dpi=100,facecolor='w',bbox_inches='tight')
def stats_scenario_sizing():
    global sorted_annual_df_operation_sizing_scenario
    statistics_table = {}
    mean_table =  defaultdict(list)
    std_table =   defaultdict(list)
    CV_table =   defaultdict(list)
    cost_points= defaultdict(list)
    emissions_points=defaultdict(list)
    label_points=defaultdict(lambda: defaultdict(list))
    sorted_cost = []
    output_prefix_short ={}
    for scenario in range(len(epw_names)):
        output_prefix =   'total_'+epw_names[scenario]+'_'
        annual_df_operation_sizing_scenario[output_prefix]=annual_df_operation_sizing_scenario[output_prefix].sort_values('Cost ($)')
        annual_df_operation_sizing_scenario[output_prefix]=annual_df_operation_sizing_scenario[output_prefix].reset_index()
        sorted_cost.append(annual_df_operation_sizing_scenario[output_prefix]['Cost ($)'][0]/10**6)
    sorted_cost = sorted(sorted_cost)
    sorted_annual_df_operation_sizing_scenario = {}
    for i in sorted_cost:
        for scenario in range(len(epw_names)):
            output_prefix =   'total_'+epw_names[scenario]+'_'
            if annual_df_operation_sizing_scenario[output_prefix]['Cost ($)'][0]/10**6 == i:
                sorted_annual_df_operation_sizing_scenario[output_prefix] =annual_df_operation_sizing_scenario[output_prefix]

    cols = ['Boilers Capacity (kW)', 'CHP Electricty Capacity (kW)', 'Battery Capacity (kW)','Solar Area (m^2)','Swept Area (m^2)','Emission (kg CO2)','Cost ($)']
    cols_revised = ['Boilers (kW)', 'CHP (kW)', 'Battery (kW)','Solar (m^2)','Wind (m^2)','Emissions (million ton)','Cost (million \$)']
    for point in range(population_size_sizing):
        for scenario in range(len(epw_names)):
            output_prefix = 'total_'+epw_names[scenario]+'_'
            cost_points[point].append(sorted_annual_df_operation_sizing_scenario[output_prefix]['Cost ($)'][point])
            emissions_points[point].append(sorted_annual_df_operation_sizing_scenario[output_prefix]['Emission (kg CO2)'][point])
            for component in cols:
                label_points[point][component].append(sorted_annual_df_operation_sizing_scenario[output_prefix][component][point])
    for point in range(population_size_sizing):
        for component in cols:
            if len(label_points[point][component])!=0:
                if component=='Emission (kg CO2)' or component=='Cost ($)':
                    std_table[point].append(round(statistics.stdev(label_points[point][component])/10**6,2))
                    mean_table[point].append(round(np.mean(label_points[point][component])/10**6,2))
                else:
                    std_table[point].append(round(statistics.stdev(label_points[point][component]),2))
                    mean_table[point].append(round(np.mean(label_points[point][component]),2))
                if np.mean(label_points[point][component])!=0:
                    CV_table[point].append(round(statistics.stdev(label_points[point][component])*100/np.mean(label_points[point][component]),2))
                else:
                    CV_table[point].append(0)

    statistics_table = {'Mean PP1': mean_table[0], 'STD  PP1': std_table[0], 'CV \% PP1': CV_table[0],
    'Mean medium cost': mean_table[24], 'STD medium cost': std_table[24], 'CV \% PP5': CV_table[24],
    'Mean max cost': mean_table[49], 'STD max cost': std_table[49], 'CV \% PP9': CV_table[49]
    }

    df_statistics_table= pd.DataFrame(statistics_table)
    df_statistics_table.insert(0, "", cols_revised, True)
    for i in range(1,len(df_statistics_table.columns)*2):
        if i%2!=0:
            df_statistics_table.insert(i, "&", ["&"]*len(df_statistics_table), True)

    df_statistics_table.insert(i, "\\\\ \hline", ["\\\\ \hline"]*len(df_statistics_table), True)
    df_statistics_table.to_csv(os.path.join(results_compare,'stats_scenario_sizing_table.csv'))
def stats_main_sizing():
    global sorted_annual_df_operation_sizing_main
    statistics_table = {}
    mean_table =  defaultdict(list)
    std_table =   defaultdict(list)
    CV_table =   defaultdict(list)
    cost_points= defaultdict(list)
    emissions_points=defaultdict(list)
    label_points=defaultdict(lambda: defaultdict(list))
    sorted_cost = []
    output_prefix_short =[]
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            output_prefix_short.append(dict_EPWs_bar_names[key][dict_EPWs[key].index(epw_file_name)])
            annual_df_operation_sizing_main[output_prefix]=annual_df_operation_sizing_main[output_prefix].sort_values('Cost ($)')
            annual_df_operation_sizing_main[output_prefix]=annual_df_operation_sizing_main[output_prefix].reset_index()
            sorted_cost.append(annual_df_operation_sizing_main[output_prefix]['Cost ($)'][0]/10**6)
    sorted_cost = sorted(sorted_cost)
    sorted_annual_df_operation_sizing_main = {}
    for i in sorted_cost:
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                if annual_df_operation_sizing_main[output_prefix]['Cost ($)'][0]/10**6 == i:
                    sorted_annual_df_operation_sizing_main[output_prefix] =annual_df_operation_sizing_main[output_prefix]

    cols = ['Boilers Capacity (kW)', 'CHP Electricty Capacity (kW)', 'Battery Capacity (kW)','Solar Area (m^2)','Swept Area (m^2)','Emission (kg CO2)','Cost ($)']
    cols_revised = ['Boilers (kW)', 'CHP (kW)', 'Battery (kW)','Solar (m^2)','Wind (m^2)','Emissions (million ton)','Cost (million \$)']
    for point in range(population_size_sizing):
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                cost_points[point].append(sorted_annual_df_operation_sizing_main[output_prefix]['Cost ($)'][point])
                emissions_points[point].append(sorted_annual_df_operation_sizing_main[output_prefix]['Emission (kg CO2)'][point])
                for component in cols:
                    label_points[point][component].append(sorted_annual_df_operation_sizing_main[output_prefix][component][point])
    SMALL_SIZE = 22
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28
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
    plt.rcParams["figure.figsize"] = (35,15)
    for component in cols:
        data_1 = []
        data_2 = []
        data_3 = []
        if component=='Emission (kg CO2)' or component=='Cost ($)':
            for key in dict_EPWs.keys():
                for epw_file_name in dict_EPWs[key]:
                    output_prefix =  'total_'+epw_file_name+'_'
                    data_1.append(sorted_annual_df_operation_sizing_main[output_prefix][component][0]/10**6)
                    data_2.append(sorted_annual_df_operation_sizing_main[output_prefix][component][24]/10**6)
                    data_3.append(sorted_annual_df_operation_sizing_main[output_prefix][component][49]/10**6)
            if component=='Emission (kg CO2)':
                component='Emissions (million ton)'
            elif component=='Cost ($)':
                component='Cost (million \$)'
        else:
            for key in dict_EPWs.keys():
                for epw_file_name in dict_EPWs[key]:
                    output_prefix =  'total_'+epw_file_name+'_'
                    data_1.append(sorted_annual_df_operation_sizing_main[output_prefix][component][0])
                    data_2.append(sorted_annual_df_operation_sizing_main[output_prefix][component][24])
                    data_3.append(sorted_annual_df_operation_sizing_main[output_prefix][component][49])
        data = [data_1,data_2,data_3]
        index = output_prefix_short
        df = pd.DataFrame({'min cost': data[0],
                           'med cost': data[1],
                           'max cost':data[2] }, index=index)

        ax = df.plot.bar(rot=0)
        ax.set_xlabel('Weather Files')
        ax.set_ylabel(component)
        ax.figure.savefig(os.path.join(results_compare,component.replace('\\$','')+'_bar_main.png'),bbox_inches='tight')

    for point in range(population_size_sizing):
        for component in cols:
            if len(label_points[point][component])!=0:
                if component=='Emission (kg CO2)' or component=='Cost ($)':
                    #print(point,round(np.mean(label_points[point]['Cost ($)'])/10**6,2))
                    std_table[point].append(round(statistics.stdev(label_points[point][component])/10**6,2))
                    mean_table[point].append(round(np.mean(label_points[point][component])/10**6,2))
                else:
                    std_table[point].append(round(statistics.stdev(label_points[point][component]),2))
                    mean_table[point].append(round(np.mean(label_points[point][component]),2))
                if np.mean(label_points[point][component])!=0:
                    CV_table[point].append(round(statistics.stdev(label_points[point][component])*100/np.mean(label_points[point][component]),2))
                else:
                    CV_table[point].append(0)

    statistics_table = {'Mean PP1': mean_table[0], 'STD  PP1': std_table[0], 'CV \% PP1': CV_table[0],
    'Mean medium cost': mean_table[24], 'STD medium cost': std_table[24], 'CV \% PP5': CV_table[24],
    'Mean max cost': mean_table[49], 'STD max cost': std_table[49], 'CV \% PP9': CV_table[49]}

    df_statistics_table= pd.DataFrame(statistics_table)
    df_statistics_table.insert(0, "", cols_revised, True)
    for i in range(1,len(df_statistics_table.columns)*2):
        if i%2!=0:
            df_statistics_table.insert(i, "&", ["&"]*len(df_statistics_table), True)

    df_statistics_table.insert(i, "\\\\ \hline", ["\\\\ \hline"]*len(df_statistics_table), True)
    df_statistics_table.to_csv(os.path.join(results_compare,'stats_main_sizing_table.csv'))


sizing()
#main_paretofront_sizing()
#scenario_paretofront_sizing()
stats_scenario_sizing()
stats_main_sizing()

### Operation Planning of DES ###
def operation_planning():
    global df_object_OP_main,df_operation_OP_main,annual_df_object_OP_main,annual_df_operation_OP_main
    df_object_OP_main= defaultdict(list)
    df_operation_OP_main = defaultdict(list)
    annual_df_object_OP_main= {}
    annual_df_operation_OP_main = {}
    j = 0
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix = 'total_'+epw_file_name+'_'
            for represent in range(num_clusters):
                data_represent =  pd.read_csv(os.path.join(representative_days_path,output_prefix + 'Represent_days_modified_'+str(represent)+ '.csv'))
                file_name = output_prefix+city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_EA_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
                results_path = os.path.join(path_test,'Operation_results', file_name)
                df_object_OP_main[output_prefix].append(pd.read_csv(os.path.join(results_path ,output_prefix+'_'+ str(represent)+'_objectives.csv')).dropna()*weight_representative_day_main[output_prefix][represent][0])
                df_operation_OP_main[output_prefix].append(pd.read_csv(os.path.join(results_path,output_prefix+'_'+ str(represent)+'_sizing.csv')).dropna()*weight_representative_day_main[output_prefix][represent][0])
                if represent !=0:
                    df_object_OP_main[output_prefix][represent] = df_object_OP_main[output_prefix][represent].add(df_object_OP_main[output_prefix][represent-1])
                    df_operation_OP_main[output_prefix][represent] = df_operation_OP_main[output_prefix][represent].add(df_operation_OP_main[output_prefix][represent-1])
            annual_df_object_OP_main[output_prefix] = df_object_OP_main[output_prefix][represent]
            annual_df_operation_OP_main[output_prefix] = df_operation_OP_main[output_prefix][represent]
            j = j+1
    j=0
    global df_object_OP_scenario,df_operation_OP_scenario,annual_df_object_OP_scenario,annual_df_operation_OP_scenario
    df_object_OP_scenario= defaultdict(list)
    df_operation_OP_scenario = defaultdict(list)
    annual_df_object_OP_scenario= {}
    annual_df_operation_OP_scenario = {}
    for scenario in range(len(epw_names)):
        output_prefix =   'total_'+epw_names[scenario]+'_'
        for represent in range(num_clusters):
            data_represent =  pd.read_csv(os.path.join(representative_days_path,output_prefix + 'Represent_days_modified_'+str(represent)+ '.csv'))
            file_name = output_prefix+city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_EA_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
            results_path = os.path.join(path_test,'Operation_results', file_name)
            df_object_OP_scenario[output_prefix].append(pd.read_csv(os.path.join(results_path ,output_prefix+'_'+ str(represent)+'_objectives.csv')).dropna()*weight_representative_day_scenario[output_prefix][represent][0])
            df_operation_OP_scenario[output_prefix].append(pd.read_csv(os.path.join(results_path,output_prefix+'_'+ str(represent)+'_sizing.csv')).dropna()*weight_representative_day_scenario[output_prefix][represent][0])
            if represent !=0:
                df_object_OP_scenario[output_prefix][represent] = df_object_OP_scenario[output_prefix][represent].add(df_object_OP_scenario[output_prefix][represent-1])
                df_operation_OP_scenario[output_prefix][represent] = df_operation_OP_scenario[output_prefix][represent].add(df_operation_OP_scenario[output_prefix][represent-1])
        annual_df_object_OP_scenario[output_prefix] = df_object_OP_scenario[output_prefix][represent]
        annual_df_operation_OP_scenario[output_prefix] = df_operation_OP_scenario[output_prefix][represent]
def paretofront_OP_main():
    import itertools
    SMALL_SIZE = 20
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 28
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
    plt.rcParams["figure.figsize"] = (22,15)
    plt.figure()
    j=0
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                 for i in range(27)]
    sorted_cost = []
    output_prefix_short ={}
    marker = itertools.cycle(('v','+','s','^', 'o','x','*'))
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            output_prefix_short[output_prefix] =  dict_EPWs_names[key][dict_EPWs[key].index(epw_file_name)]
            if key is 'AMYs':
                year_selected_number=int(output_prefix_short[output_prefix].replace('AMY',''))
                if year_selected_number==2019 or year_selected_number==2018 or year_selected_number==2016 or year_selected_number==2014 or year_selected_number==2012:
                    sorted_cost.append(annual_df_object_OP_main[output_prefix]['Cost ($)'][0]/10**3)
            elif key is 'TMYs':
                if epw_file_name=='USA_UT_Salt.Lake.City.Intl.AP.725720_TMY3':
                    sorted_cost.append(annual_df_object_OP_main[output_prefix]['Cost ($)'][0]/10**3)
            else:
                sorted_cost.append(annual_df_object_OP_main[output_prefix]['Cost ($)'][0]/10**3)
    sorted_cost = sorted(sorted_cost)
    sorted_annual_df_object_OP_main = {}
    for i in sorted_cost:
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                if annual_df_object_OP_main[output_prefix]['Cost ($)'][0]/10**3 == i:
                    sorted_annual_df_object_OP_main[output_prefix] =annual_df_object_OP_main[output_prefix]
    j=0
    for key in sorted_annual_df_object_OP_main:
            output_prefix =  key
            cost = [i/10**3 for i in sorted_annual_df_object_OP_main[output_prefix]['Cost ($)']]
            emissions = [j/10**3 for j in sorted_annual_df_object_OP_main[output_prefix]['Emission (kg CO2)']]
            label = output_prefix_short[output_prefix]
            plt.scatter(cost,emissions,c=color[j], s=100, cmap=cmap, label = label,marker=next(marker))
            #plt.title('Cost and emissions trade-off')
            plt.xlabel("Cost (thousand $)")
            plt.ylabel("Emissions (metric ton $CO_2$)")
            if j%2==0:
                plt.annotate(label,xy=(cost[-1], emissions[-1]),xytext=(cost[-1]*1.01, emissions[-1]*0.995),
                arrowprops=dict(arrowstyle="->"))
            elif j%2==1:
                plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.95, emissions[0]*1.005),
                arrowprops=dict(arrowstyle="->"))

            j = j+1

    j=0
    #plt.legend()
    plt.savefig(os.path.join(results_compare ,'ParetoFront_OP.png'),dpi=100,facecolor='w',bbox_inches='tight')
def paretofront_OP_scenario():
    SMALL_SIZE = 22
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28
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
    plt.rcParams["figure.figsize"] = (30,15)
    plt.figure()
    j=0

    sorted_cost = []
    output_prefix_short ={}
    sorted_cost_scenario = []
    for scenario in range(len(epw_names)):
        output_prefix =   'total_'+epw_names[scenario]+'_'
        annual_df_object_OP_scenario[output_prefix]=annual_df_object_OP_scenario[output_prefix].sort_values('Cost ($)')
        annual_df_object_OP_scenario[output_prefix]=annual_df_object_OP_scenario[output_prefix].reset_index()
        sorted_cost_scenario.append(annual_df_object_OP_scenario[output_prefix]['Cost ($)'][0]/10**3)
    sorted_cost_scenario = sorted(sorted_cost_scenario)
    sorted_annual_df_object_OP_scenario = {}
    for i in sorted_cost_scenario:
        for scenario in range(len(epw_names)):
            output_prefix =   'total_'+epw_names[scenario]+'_'
            label = output_prefix.replace('_','').replace('total','')
            if annual_df_object_OP_scenario[output_prefix]['Cost ($)'][0]/10**3 == i and (int(label[1])==0 or int(label[1])==5 or int(label[1])==9) and (int(label[3])==0 or int(label[3])==5 or int(label[3])==9):
                sorted_annual_df_object_OP_scenario[output_prefix] =annual_df_object_OP_scenario[output_prefix]

    j=0
    marker_list = ['v','+','s','^','o','x','*','s','>','<']
    color_list=  [ 'tab:blue', 'tab:orange','tab:green','black','yellow','tab:red','tab:cyan','tab:olive','peru','tab:purple']

    for key in sorted_annual_df_object_OP_scenario:
        output_prefix =  key
        cost = [i/10**3 for i in sorted_annual_df_object_OP_scenario[output_prefix]['Cost ($)']]
        emissions = [j/10**3 for j in sorted_annual_df_object_OP_scenario[output_prefix]['Emission (kg CO2)']]
        label = key.replace('_','').replace('total','')
        int_marker= int(label[1])
        int_color = int(label[3])
        #print(int_marker, type(int_marker), marker[int_marker],len( marker))
        marker = marker_list[int_marker]
        color = color_list[int_color]
        plt.scatter(cost,emissions,c=color, s=100, cmap=cmap, label = label,marker=marker)
        #plt.title('Cost and emissions trade-off')
        plt.xlabel("Cost (million $)")
        plt.ylabel("Emissions (million kg $CO_2$)")
        if int(label[1])==0:
            label_T='Tmin'
        elif int(label[1])==5:
            label_T='Tmed'
        elif int(label[1])==9:
            label_T='Tmax'
        if int(label[3])==0:
            label_S='Smin'
        elif int(label[3])==5:
            label_S='Smed'
        elif int(label[3])==9:
            label_S='Smax'
        label =  label_T + label_S
        plt.annotate(label,xy=(cost[0], emissions[0]),xytext=(cost[0]*0.96, emissions[0]*1.005),
        arrowprops=dict(arrowstyle="->"),fontsize=MEDIUM_SIZE)

    #plt.legend()
    plt.savefig(os.path.join(results_compare ,'scenario_ParetoFront_OP.png'),dpi=100,facecolor='w',bbox_inches='tight')
def stats_scenario_operation_planning():
    statistics_table = {}
    mean_table =  defaultdict(list)
    std_table =   defaultdict(list)
    CV_table =   defaultdict(list)
    cost_points= defaultdict(list)
    emissions_points=defaultdict(list)
    label_points=defaultdict(lambda: defaultdict(list))
    cols = ['Boilers Operation (kWh)', 'CHP Operation (kWh)', 'Battery Operation (kWh)','Grid Operation (kWh)','Solar Generation (kWh)','Wind Generation (kWh)','Emission (kg CO2)','Cost ($)']
    cols_revised = ['Boilers (MWh)', 'CHP (MWh)', 'Battery (MWh)','Grid (MWh)','Solar (MWh)','Wind (MWh)','Emissions (metric ton)','Cost (thousand \$)']
    for point in range(population_size):
        for scenario in range(len(epw_names)):
            output_prefix = 'total_'+epw_names[scenario]+'_'
            cost_points[point].append(annual_df_operation_OP_scenario[output_prefix]['Cost ($)'][point])
            emissions_points[point].append(annual_df_operation_OP_scenario[output_prefix]['Emission (kg CO2)'][point])
            for component in cols:
                label_points[point][component].append(annual_df_operation_OP_scenario[output_prefix][component][point])
    for point in range(population_size):
        for component in cols:
            if len(label_points[point][component])!=0:
                std_table[point].append(round(statistics.stdev(label_points[point][component])/1000,2))
                mean_table[point].append(round(np.mean(label_points[point][component])/1000,2))
                if np.mean(label_points[point][component])!=0:
                    CV_table[point].append(round(statistics.stdev(label_points[point][component])*100/np.mean(label_points[point][component]),2))
                else:
                    CV_table[point].append(0)

    statistics_table = {'Mean PP1': mean_table[0], 'STD  PP1': std_table[0], 'CV \% PP1': CV_table[0],
    'Mean medium cost': mean_table[5], 'STD medium cost': std_table[5], 'CV \% PP5': CV_table[5],
    'Mean max cost': mean_table[10], 'STD max cost': std_table[10], 'CV \% PP9': CV_table[10]
    }

    df_statistics_table= pd.DataFrame(statistics_table)
    df_statistics_table.insert(0, "", cols_revised, True)
    for i in range(1,len(df_statistics_table.columns)*2):
        if i%2!=0:
            df_statistics_table.insert(i, "&", ["&"]*len(df_statistics_table), True)

    df_statistics_table.insert(i, "\\\\ \hline", ["\\\\ \hline"]*len(df_statistics_table), True)
    df_statistics_table.to_csv(os.path.join(results_compare,'stats_scenario_operation_table.csv'))
def stats_main_operation_planning():
    statistics_table = {}
    mean_table =  defaultdict(list)
    std_table =   defaultdict(list)
    CV_table =   defaultdict(list)
    cost_points= defaultdict(list)
    emissions_points=defaultdict(list)
    label_points=defaultdict(lambda: defaultdict(list))
    cols = ['Boilers Operation (kWh)', 'CHP Operation (kWh)', 'Battery Operation (kWh)','Grid Operation (kWh)','Solar Generation (kWh)','Wind Generation (kWh)','Emission (kg CO2)','Cost ($)']
    cols_revised = ['Boilers (MWh)', 'CHP (MWh)', 'Battery (MWh)','Grid (MWh)','Solar (MWh)','Wind (MWh)','Emissions (metric ton)','Cost (thousand \$)']
    for point in range(population_size):
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                cost_points[point].append(annual_df_operation_OP_main[output_prefix]['Cost ($)'][point])
                emissions_points[point].append(annual_df_operation_OP_main[output_prefix]['Emission (kg CO2)'][point])
                for component in cols:
                    label_points[point][component].append(annual_df_operation_OP_main[output_prefix][component][point])

    for point in range(population_size):
        for component in cols:
            if len(label_points[point][component])!=0:
                std_table[point].append(round(statistics.stdev(label_points[point][component])/1000,2))
                mean_table[point].append(round(np.mean(label_points[point][component])/1000,2))
                if np.mean(label_points[point][component])!=0:
                    CV_table[point].append(round(statistics.stdev(label_points[point][component])*100/np.mean(label_points[point][component]),2))
                else:
                    CV_table[point].append(0)

    statistics_table = {'Mean PP1': mean_table[0], 'STD  PP1': std_table[0], 'CV \% PP1': CV_table[0],
    'Mean medium cost': mean_table[5], 'STD medium cost': std_table[5], 'CV \% PP5': CV_table[5],
    'Mean max cost': mean_table[10], 'STD max cost': std_table[10], 'CV \% PP9': CV_table[10]
    }

    df_statistics_table= pd.DataFrame(statistics_table)
    df_statistics_table.insert(0, "", cols_revised, True)
    for i in range(1,len(df_statistics_table.columns)*2):
        if i%2!=0:
            df_statistics_table.insert(i, "&", ["&"]*len(df_statistics_table), True)

    df_statistics_table.insert(i, "\\\\ \hline", ["\\\\ \hline"]*len(df_statistics_table), True)
    df_statistics_table.to_csv(os.path.join(results_compare,'stats_main_operation_table.csv'))
    global sorted_annual_df_operation_OP_main
    cost_points= defaultdict(list)
    emissions_points=defaultdict(list)
    sorted_cost = []
    output_prefix_short =[]
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            output_prefix_short.append(dict_EPWs_bar_names[key][dict_EPWs[key].index(epw_file_name)])
            annual_df_operation_OP_main[output_prefix]=annual_df_operation_OP_main[output_prefix].sort_values('Cost ($)')
            annual_df_operation_OP_main[output_prefix]=annual_df_operation_OP_main[output_prefix].reset_index()
            sorted_cost.append(annual_df_operation_OP_main[output_prefix]['Cost ($)'][0]/10**6)
    sorted_cost = sorted(sorted_cost)
    sorted_annual_df_operation_OP_main = {}
    for i in sorted_cost:
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                if annual_df_operation_OP_main[output_prefix]['Cost ($)'][0]/10**6 == i:
                    sorted_annual_df_operation_OP_main[output_prefix] =annual_df_operation_OP_main[output_prefix]

    for point in range(population_size):
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                cost_points[point].append(sorted_annual_df_operation_OP_main[output_prefix]['Cost ($)'][point])
                emissions_points[point].append(sorted_annual_df_operation_OP_main[output_prefix]['Emission (kg CO2)'][point])
                for component in cols:
                    label_points[point][component].append(sorted_annual_df_operation_OP_main[output_prefix][component][point])
    sorted_cost = []
    output_prefix_short =[]
    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            output_prefix =  'total_'+epw_file_name+'_'
            output_prefix_short.append(dict_EPWs_bar_names[key][dict_EPWs[key].index(epw_file_name)])
            annual_df_operation_OP_main[output_prefix]=annual_df_operation_OP_main[output_prefix].sort_values('Cost ($)')
            annual_df_operation_OP_main[output_prefix]=annual_df_operation_OP_main[output_prefix].reset_index()
            sorted_cost.append(annual_df_operation_OP_main[output_prefix]['Cost ($)'][0]/10**6)
    sorted_cost = sorted(sorted_cost)
    sorted_annual_df_operation_OP_main = {}
    for i in sorted_cost:
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                if annual_df_operation_OP_main[output_prefix]['Cost ($)'][0]/10**6 == i:
                    sorted_annual_df_operation_OP_main[output_prefix] =annual_df_operation_OP_main[output_prefix]

    for point in range(population_size):
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                cost_points[point].append(sorted_annual_df_operation_OP_main[output_prefix]['Cost ($)'][point])
                emissions_points[point].append(sorted_annual_df_operation_OP_main[output_prefix]['Emission (kg CO2)'][point])
                for component in cols:
                    label_points[point][component].append(sorted_annual_df_operation_OP_main[output_prefix][component][point])
    SMALL_SIZE = 22
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28
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
    plt.rcParams["figure.figsize"] = (35,15)
    cols = ['Boilers Operation (kWh)', 'CHP Operation (kWh)', 'Battery Operation (kWh)','Grid Operation (kWh)','Solar Generation (kWh)','Wind Generation (kWh)','Emission (kg CO2)','Cost ($)']
    #cols_revised = ['Boilers (MWh)', 'CHP (MWh)', 'Battery (MWh)','Grid (MWh)','Solar (MWh)','Wind (MWh)','Emissions (metric ton)','Cost (thousand \$)']
    cols_revised = ['Boilers (kWh)', 'CHP (kWh)', 'Battery (kWh)','Grid (kWh)','Solar (kWh)','Wind (kWh)','Emissions (metric ton)','Cost (thousand \$)']

    for component in cols:
        data_1 = []
        data_2 = []
        data_3 = []
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                output_prefix =  'total_'+epw_file_name+'_'
                data_1.append(sorted_annual_df_operation_OP_main[output_prefix][component][0]/10**3)
                data_2.append(sorted_annual_df_operation_OP_main[output_prefix][component][5]/10**3)
                data_3.append(sorted_annual_df_operation_OP_main[output_prefix][component][10]/10**3)

        data = [data_1,data_2,data_3]
        index = output_prefix_short
        df = pd.DataFrame({'min cost': data[0],
                           'med cost': data[1],
                           'max cost':data[2] }, index=index)

        ax = df.plot.bar(rot=0)
        ax.set_xlabel('Weather Files')
        component_name =  cols_revised[cols.index(component)]
        ax.set_ylabel(component_name)
        ax.figure.savefig(os.path.join(results_compare,component_name.replace('\\$','')+'_bar_main.png'),bbox_inches='tight')

representative_day_function()
operation_planning()
#paretofront_OP_scenario()
#paretofront_OP_main()
stats_scenario_operation_planning()
stats_main_operation_planning()
