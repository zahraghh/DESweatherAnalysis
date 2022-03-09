import pandas as pd
import csv
import math
import datetime as dt
import os
import sys
import csv
import statistics
import matplotlib.pyplot as plt
from sko.GA import GA

def selecting_buildings_EP(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    dict_EPWs = {}
    dict_EPWs['FMYs']=['USA_Salt Lake City Intl AP_HadCM3-A2-'+str(2080),'USA_Salt Lake City Intl AP_HadCM3-A2-'+str(2050)]
    dict_EPWs['TMYs']=['USA_UT_Salt.Lake.City.Intl.AP.725720_TMY','USA_UT_Salt.Lake.City.725720_TMY2','USA_UT_Salt.Lake.City.Intl.AP.725720_TMY3']
    weather_file= dict_EPWs['TMYs'][-1]
    year=2019
    city=editable_data['city']
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
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
    path_test =  os.path.join(sys.path[0])
    energy_demands_path = os.path.join(path_test,'Buildings Energy Demands')
    prototype_energyplus_path = os.path.join(path_test,'IDFBuildingsFiles')
    #energyplus_buildings = ['ApartmentHighRise', 'ApartmentMidRise','Hospital','HotelLarge','HotelSmall','OfficeLarge','OfficeMedium','OfficeSmall','OutPatientHealthCare','RestaurantFastFood','RestaurantSitDown','RetailStandAlone','RetailStripMall','SchoolPrimary','SchoolSecondary','Warehouse']
    #energyplus_buildings = ['OfficeLarge','OfficeMedium','OfficeSmall','OutPatientHealthCare','SchoolPrimary','SchoolSecondary','Warehouse']
    #energyplus_buildings = ['OfficeMedium','OutPatientHealthCare','RetailStandAlone','SchoolSecondary']
    energyplus_buildings = ['OfficeMedium','OfficeMedium','Hospital','RetailStandAlone']
    thermal_eff_dict = {'OfficeMedium':0.8,'Hospital':0.8125,'RetailStandAlone':0.82}

    energyplus_buildings_rev = []
    UoU_final = [ 'SFEBB', 'HEB','EIHG', 'SLC'] #in 2019
    JtokWh = 0.27777777777778/10**6
    dict_UoU = {}
    dict_energyplus = {}
    for UoU in UoU_final:
        dict_UoU[UoU] = pd.read_csv(os.path.join(energy_demands_path,UoU+'_processed.csv'),index_col=0)
        #plt.plot(dict_UoU[UoU]['Electricity (kWh)'], label=UoU)
        plt.plot(dict_UoU[UoU]['Heating (kWh)'], label=UoU)
    num = 1
    for energyplus in energyplus_buildings:
        dict_energyplus[energyplus+'_'+str(num)] = pd.read_csv(os.path.join(prototype_energyplus_path ,'ASHRAE901_'+energyplus+'_STD2019_Denver_'+weather_file+'_mtr.csv'),index_col=0)*JtokWh
        energyplus_buildings_rev.append(energyplus+'_'+str(num))
        #Gas:Facility[J](Hourly) and Heating:Electricity [J](Hourly)
        #print(dict_energyplus[energyplus+'_'+str(num)].keys())
        if 'Electricity:Facility [J](Hourly)' in dict_energyplus[energyplus+'_'+str(num)].keys():
            dict_energyplus[energyplus+'_'+str(num)]['Electricity kWh'] = dict_energyplus[energyplus+'_'+str(num)]['Electricity:Facility [J](Hourly)'] - dict_energyplus[energyplus+'_'+str(num)]['Heating:Electricity [J](Hourly)']
            #plt.plot(dict_energyplus[energyplus+'_'+str(num)][''Electricity kWh'], label=[energyplus+'_'+str(num)])
        if 'Gas:Facility [J](Hourly)' in dict_energyplus[energyplus+'_'+str(num)].keys() and 'Heating:Electricity [J](Hourly)' in dict_energyplus[energyplus+'_'+str(num)].keys():
            dict_energyplus[energyplus+'_'+str(num)]['Heating kWh'] = dict_energyplus[energyplus+'_'+str(num)]['Gas:Facility [J](Hourly)']*thermal_eff_dict[energyplus] + dict_energyplus[energyplus+'_'+str(num)]['Heating:Electricity [J](Hourly)']
            plt.plot(dict_energyplus[energyplus+'_'+str(num)]['Heating kWh'], label=[energyplus+'_'+str(num)])
        print(energyplus, sum(dict_energyplus[energyplus+'_'+str(num)]['Electricity kWh'])/1000,sum(dict_energyplus[energyplus+'_'+str(num)]['Heating kWh'])/1000 )
        num = num + 1

    plt.xticks(list(range(1,8760,730)),list(range(1,13)))
    plt.legend()
    #plt.savefig(os.path.join(path_test,'Electricity_total'+'.png'),dpi=300,facecolor='w')
    plt.savefig(os.path.join(path_test,'Heating_total'+'.png'),dpi=300,facecolor='w')
    plt.close()
    getting_WFs = 'yes'
    if getting_WFs == 'yes':
        best_x=[]
        best_y=[]
        for i in range(4):
            def Obj_GA(p):
                weight_factor= p
                MSE_elect =((dict_energyplus[energyplus_buildings_rev[i]]['Electricity kWh']*weight_factor).array - dict_UoU[UoU_final[i]]['Electricity (kWh)'].array)**2
                MSE_heating = ((dict_energyplus[energyplus_buildings_rev[i]]['Heating kWh']*weight_factor).array - dict_UoU[UoU_final[i]]['Heating (kWh)'].array)**2
                return (sum(MSE_elect)+sum(MSE_heating))/(8760*2)
            #print(energyplus_buildings[i],UoU_final[i])
            GA_model = GA(func=Obj_GA, n_dim=1, size_pop=50, max_iter=800, lb=[0],)
            results = GA_model.run()
            best_x.append(results[0])
            best_y.append(results[1])
            print(results[0],results[1])
        print('weight_factor',best_x)
        print('MSE',best_y)
    weight_factor_2004 = [0.50558832,1.0,0.23360898,1.0]
    RSME_WFs_2004 = [4927.3237312**0.5,7279.71216085**0.5,29176.72420875**0.5,4590.48390218**0.5] #kWh
    weight_factor = [0.50558832,1.0,0.35786005,1.0]
    RSME_WFs = [4894.322211714833**0.5,9010.63054282**0.5,27487.01264646**0.5,6030.52241506**0.5] #kWh
    mean_error_WFs = []
    for i in range(4):
        mean_error_WFs.append(RSME_WFs[i]/statistics.mean(dict_energyplus[energyplus_buildings_rev[i]]['Electricity kWh']+dict_energyplus[energyplus_buildings_rev[i]]['Heating kWh']))
        dict_energyplus[energyplus_buildings_rev[i]]['Electricity kWh']=dict_energyplus[energyplus_buildings_rev[i]]['Electricity kWh']*weight_factor[i]
        dict_energyplus[energyplus_buildings_rev[i]]['Heating kWh']=dict_energyplus[energyplus_buildings_rev[i]]['Heating kWh']*weight_factor[i]
    print(mean_error_WFs)
    return dict_energyplus
path_test =  os.path.join(sys.path[0])

selecting_buildings_EP(path_test)
