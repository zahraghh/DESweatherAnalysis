import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import plotly.express as px
import os
import sys
import json
import pandas as pd
from pathlib import Path
from nested_dict import nested_dict
from collections import defaultdict
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
representative_days_path= os.path.join(path_test,'ScenarioReduction')
demand_directory = os.path.join(path_test, 'IDFBuildingsFiles')
df_object = {}
df_operation = {}
dict_EPWs={}
num_scenarios = int(editable_data['num_scenarios'])
num_clusters = int(editable_data['Cluster numbers'])+2
city =str(editable_data['city'])
JtokWh = 2.7778e-7
weight_factor =  [1.50558832,0.35786005,1.0]
idf_names = ['ASHRAE901_OfficeMedium_STD2019_Denver','ASHRAE901_Hospital_STD2019_Denver','ASHRAE901_RetailStandalone_STD2019_Denver']
thermal_eff_dict = {idf_names[0]:0.8,idf_names[1]:0.8125,idf_names[2]:0.82}
dict_EPWs['TMYs']=['USA_UT_Salt.Lake.City.Intl.AP.725720_TMY','USA_UT_Salt.Lake.City.725720_TMY2','USA_UT_Salt.Lake.City.Intl.AP.725720_TMY3']
total_electricity_buildings = []
total_heating_buildings = []
energy_demands = {}
represent_day_path = {}
for building_type in idf_names:
    output_prefix =  building_type+'_'+dict_EPWs['TMYs'][-1]+'_mtr.csv'
    demand_data_path = os.path.join(demand_directory, output_prefix)
    data = pd.read_csv(demand_data_path)
    elect_data = (data['Electricity:Facility [J](Hourly)']-data['Heating:Electricity [J](Hourly)'])*JtokWh
    heat_data = (data['Gas:Facility [J](Hourly)']*thermal_eff_dict[building_type]+data['Heating:Electricity [J](Hourly)'])*JtokWh
    total_electricity_buildings.append(elect_data*weight_factor[idf_names.index(building_type)])
    total_heating_buildings.append(heat_data*weight_factor[idf_names.index(building_type)])

name_total =  'total_'+dict_EPWs['TMYs'][-1]+'_'
represent_day = {}
for day in range(num_clusters):
    represent_day[day] =  pd.read_csv(os.path.join(representative_days_path, name_total + 'Represent_days_modified_'+str(day)+ '.csv'))

energy_demands['Total Electricity (kWh)']=sum(total_electricity_buildings)
energy_demands['Total Heating (kWh)']=sum(total_heating_buildings)
energy_demands_daily= defaultdict(list)
represent_day_daily= defaultdict(list)
#electricity
label_cluster = []
for day in range(365):
    plt.plot(energy_demands['Total Electricity (kWh)'][day*24:(day+1)*24].to_list(),linewidth=0.5)
    energy_demands_daily['Total Electricity (kWh)'].append(sum(energy_demands['Total Electricity (kWh)'][day*24:(day+1)*24].to_list()))
    plt.xlabel("Hour")
    plt.ylabel("Total Electricity Demand (kWh)")

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
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
plt.rcParams["figure.figsize"] = (8,6)

for i in range(num_clusters):
    label_cluster.append(represent_day[i]['Percent %'][0])
    represent_day_daily['Electricity total (kWh)'].append(sum(represent_day[i]['Electricity total (kWh)']))
    plt.plot(represent_day[i]['Electricity total (kWh)'],linewidth=5.0)
plt.savefig(os.path.join(path_test,'electricity_cluster'+'.png'),dpi=300,facecolor='w')
plt.close()
#heating
for day in range(365):
    plt.plot(energy_demands['Total Heating (kWh)'][day*24:(day+1)*24].to_list(),linewidth=0.5)
    energy_demands_daily['Total Heating (kWh)'].append(sum(energy_demands['Total Heating (kWh)'][day*24:(day+1)*24].to_list()))
    plt.xlabel('Hour')
    plt.ylabel('Total Heating Demand (kWh)')
for i in range(num_clusters):
    plt.plot(represent_day[i]['Heating (kWh)'],linewidth=5.0)
    represent_day_daily['Heating (kWh)'].append(sum(represent_day[i]['Heating (kWh)']))
plt.savefig(os.path.join(path_test,'heating_cluster'+'.png'),dpi=300,facecolor='w')
plt.close()

#Energy Demands Together
import random
number_of_colors = 365
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
for day in range(365):
    #print(len(energy_demands_daily['Total Electricity (kWh)']),len(energy_demands_daily['Total Heating (kWh)']))
    plt.scatter(energy_demands_daily['Total Electricity (kWh)'][day],energy_demands_daily['Total Heating (kWh)'][day],s=0.5, c= color[day])
    handles, labels = plt.gca().get_legend_handles_labels()
color_cluster = ['red','greenyellow','teal','blue','magenta']
label = {}
plot_combo = {}
for i in range(num_clusters):
    if label_cluster[i]>=0 and label_cluster[i]<5:
        color=color_cluster[0]
        size = 5*6
        if 0 not in label.keys():
            label[0]='0-5%'
            plot_combo[0]=plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color,label=label[0])
        else:
            plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color)
    if label_cluster[i]>=5 and label_cluster[i]<10:
        color=color_cluster[1]
        size = 10*6
        if 1 not in label.keys():
            label[1]='5-10%'
            plot_combo[1]=plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color,label=label[1])
        else:
            plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color)
    if label_cluster[i]>=10 and label_cluster[i]<15:
        color=color_cluster[2]
        size = 15*6
        if 2 not in label.keys():
            label[2]='10-15%'
            plot_combo[2]=plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color,label=label[2])
        else:
            plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color)
    if label_cluster[i]>=15 and label_cluster[i]<20:
        color=color_cluster[3]
        size = 20*6
        if 3 not in label.keys():
            label[3]='15-20%'
            plot_combo[3]=plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color,label=label[3])
        else:
            plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color)
    if label_cluster[i]>=20 and label_cluster[i]<25:
        color=color_cluster[4]
        size = 25*6
        if 4 not in label.keys():
            label[4]='20-25%'
            plot_combo[4]=plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color,label=label[4])
        else:
            plt.scatter(represent_day_daily['Electricity total (kWh)'][i],represent_day_daily['Heating (kWh)'][i],s=size, c= color)
plt.ylabel('Hot water demand (kWh)')
plt.xlabel('Electricity demand (kWh)')
#plt.legend([plot_combo[0],plot_combo[1],plot_combo[2], plot_combo[3], plot_combo[4]], [label[0],label[1],label[2], label[3], label[4]],title='Probability of representative days')
plt.legend([plot_combo[0],plot_combo[1], plot_combo[3]], [label[0],label[1], label[3]],title='Probability of representative days')
plt.savefig(os.path.join(path_test, 'enegry_demands_clusters.png'),dpi=300)
plt.show()
