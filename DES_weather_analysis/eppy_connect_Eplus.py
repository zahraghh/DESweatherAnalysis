import sys
import os
import pandas as pd
import subprocess
from pathlib import Path

def run_Eplus(path_test):
    #pathnameto_eppy = '../'
    #sys.path.append(pathnameto_eppy)
    #sys.path.append(os.path.join(path_test, 'IDFBuildingsFiles'))
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    num_scenarios = int(editable_data['num_scenarios'])
    city=editable_data['city']
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    iddfile = editable_data['iddfile']
    exefile = editable_data['exefile']
    iddfile = editable_data['iddfile']
    exefile = editable_data['exefile']
    e_plus_path = Path(exefile).parent
    #sys.path.append(e_plus_path)
    #sys.path.append(Path(path_test).parent)
    output_directory = os.path.join(path_test, 'IDFBuildingsFiles')
    weather_directory = os.path.join(path_test,'Weather files')
    #sys.path.append(output_directory)
    #sys.path.append(weather_directory)

    idf_names= []
    for i in range(int(editable_data['number_buildings'])):
        if 'building_name_'+str(i+1) in editable_data.keys():
            building_name = editable_data['building_name_'+str(i+1)]
            idf_names.append(building_name)
    #idf_names = idf_names[1:3]
    start_year = int(editable_data['starting_year'])
    end_year = int(editable_data['ending_year'])
    dict_EPWs = {}
    list_years = []
    list_tmys =[]
    list_fmys = []
    for year in range(start_year,end_year+1):
        weather_data = city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)
        list_years.append(weather_data)
    for i in range(5):
        if 'TMY'+str(i+1)+'_name' in editable_data.keys():
            TMY_name = editable_data['TMY'+str(i+1)+'_name']
            list_tmys.append(TMY_name)
        if 'FMY'+str(i+1)+'_name'  in editable_data.keys():
            FMY_name = editable_data['FMY'+str(i+1)+'_name']
            list_fmys.append(FMY_name)
    dict_EPWs['AMYs']=list_years
    dict_EPWs['FMYs']=list_fmys
    dict_EPWs['TMYs']=list_tmys
    main_weather_epw = {}
    epw_names = []
    for i_temp in range(num_scenarios):
        for i_solar in range(num_scenarios):
            epw_names.append('T_'+str(i_temp)+'_S_'+str(i_solar))

    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            if key == 'AMYs':
                epw_main_path =  os.path.join(os.path.join(path_test,'ScenarioGeneration'),epw_file_name+'.epw')
            else:
                epw_main_path = os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.epw')
            main_weather_epw[epw_file_name] =  epw_main_path
            for building_type in range(len(idf_names)):
                idf_path =  os.path.join(output_directory, idf_names[building_type]+'.idf')
                #idf = IDF(idf_path, epw_main_path)
                #output_meter = idf.idfobjects['OUTPUT:METER:METERFILEONLY']
                #timestep = idf.idfobjects['TIMESTEP']
                #timestep.Number_of_Timesteps_per_Hour = 1
                output_prefix =  idf_names[building_type]+'_'+epw_file_name+'_'
                #idf.run(output_directory=output_directory,output_prefix=output_prefix,output_suffix='C', expandobjects=True,readvars=True)
                #subprocess.call([exefile, '-w', epw_path, idf_path,'-d',output_directory,'-p',output_prefix,'-r'])
                df = subprocess.Popen([exefile, '-w', epw_main_path,'-d',output_directory,'-p',output_prefix,'-r',idf_path], stdout=subprocess.PIPE)
                output, err = df.communicate()
                print(output_prefix,' is done')


    for building_type in range(len(idf_names)):
        for scenario in range(len(epw_names)):
            idf_path =  os.path.join(output_directory, idf_names[building_type]+'.idf')
            epw_path =  os.path.join(os.path.join(path_test,'ScenarioGeneration'),epw_names[scenario]+'.epw')
            #idf = IDF(idf_path, epw_path)
            #output_meter = idf.idfobjects['OUTPUT:METER:METERFILEONLY']
            #timestep = idf.idfobjects['TIMESTEP']
            #timestep.Number_of_Timesteps_per_Hour = 1
            output_prefix =  idf_names[building_type]+'_'+epw_names[scenario]+'_'
            #idf.run(output_directory=output_directory,output_prefix=output_prefix,output_suffix='C', expandobjects=True, readvars=True)
            #subprocess.call([exefile, '-w', epw_path, idf_path,'-d',output_directory,'-p',output_prefix,'-r'])
            df = subprocess.Popen([exefile, '-w', epw_path,'-d',output_directory,'-p',output_prefix,'-r',idf_path], stdout=subprocess.PIPE)
            output, err = df.communicate()
            print(scenario,output_prefix,' is done')
