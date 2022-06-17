import datetime
import math
import time
import urllib
import pandas as pd
import numpy as np
import MesoPy as Meso
import os
import sys
import csv
import DES_weather_analysis
from DES_weather_analysis import EPW_to_csv, psychropy
# adapted from https://github.com/SSESLab/laf/blob/master/LAF.py
def read_tmy3(tmy3_name):
    if len(tmy3_name) is 0:
        nothing = 1
    else:
        ############################
        # Read TMY3 header
        ############################
        f = open(tmy3_name)
        global header
        header = []
        for i in range(0, 8):
            line = f.readline()
            header.append(line)
        f.close()

        f2 = open(tmy3_name, 'rt')
        first_line = next(csv.reader(f2))
        for i in range(0, 4):
            next(csv.reader(f2))
        comm_line = next(csv.reader(f2))
        f2.close()
        #
        global lat, long, DS, City, State, Country, elev, wmosn, comm1, tz, refy
        lat = first_line[6]
        long = first_line[7]
        DS = 'LAF'
        City = first_line[1]
        State = first_line[2]
        Country = first_line[3]
        elev = first_line[9]
        wmosn = first_line[5]
        comm1 = comm_line[1]
        tz = first_line[8]

        ############################
        # Read TMY3 data
        ############################
        data = read_datafile(tmy3_name, 8)
        global Y, M, D, HH, MM, Tdb, Tdew, RH, Patm, ExHorRad, ExDirNormRad, HorIR, GHRad, DNRad, DHRad, GHIll, DNIll, DHIll
        global HorIR, GHRad, GHIll, DNIll, DHIll, ZenLum, Wdir, Wspeed, TotSkyCover, OpSkyCover, Visib, CeilH, PrecWater
        global AerOptDepth, SnowDepth, DSLS, Albedo, LiqPrecDepth, LiqPrecQuant, PresWeathObs, PresWeathCodes

        Y = data[:, 0]
        M = data[:, 1]
        D = data[:, 2]
        HH = data[:, 3]
        MM = data[:, 4]
        Tdb = data[:, 6]
        Tdew = data[:, 7]
        RH = data[:, 8]
        Patm = data[:, 9]
        ExHorRad = data[:, 10]
        ExDirNormRad = data[:, 11]
        HorIR = data[:, 12]
        GHRad = data[:, 13]
        DNRad = data[:, 14]
        DHRad = data[:, 15]
        GHIll = data[:, 16]
        DNIll = data[:, 17]
        DHIll = data[:, 18]
        ZenLum = data[:, 19]
        Wdir = data[:, 20]
        Wspeed = data[:, 21]
        TotSkyCover = data[:, 22]
        OpSkyCover = data[:, 23]
        Visib = data[:, 24]
        CeilH = data[:, 25]
        PresWeathObs = data[:, 26]
        PresWeathCodes = data[:, 27]
        PrecWater = data[:, 28]
        AerOptDepth = data[:, 29]
        SnowDepth = data[:, 30]
        DSLS = data[:, 31]
        Albedo = data[:, 32]
        LiqPrecDepth = data[:, 33]
        LiqPrecQuant = data[:, 34]
        ############################
        # Date/Time vector
        ############################
        global DateTime
        DateTime = []
        for i in range(0, 8760):
            dt = str(int(Y[i])) + '/' + str(int(M[i])) + '/' + str(int(D[i])) + ' - ' + str(int(HH[i])) + ':' + str(int(MM[i])) + ':00'
            DateTime.append(dt)
        #return  GHRad, DNRad, DHRad, GHIll, DNIll, DHIll,  Wspeed

# adapted from https://github.com/SSESLab/laf/blob/master/LAF.py
def read_datafile(file_name, skiplines):
    data = np.genfromtxt(file_name, delimiter=',', skip_header=skiplines)
    return data

# adapted from https://github.com/SSESLab/laf/blob/master/LAF.py

def read_csv(_df):
    global GHRad, DNRad, DHRad, GHIll, DNIll, DHIll,  Wspeed, ExHorRad, Tdb, HorIR, ZenLum
    ############################
    # CSV file reading
    ############################
    Tdb = [round(i,3) for i in _df['Temperature']]
    GHRad = [round(i,3) for i in _df['GHI']]
    DNRad = [round(i,3) for i in _df['DNI']]
    DHRad = [round(i,3) for i in _df['DHI']]
    ExHorRad = [round(i,3) for i in _df['DHI_extra']]
    HorIR = [round(i,3) for i in _df['horiz_ir']]
    GHIll = [round(i,3) for i in _df['GHI_illum']]
    DNIll = [round(i,3) for i in _df['DNI_illum']]
    DHIll = [round(i,3) for i in _df['DHI_illum']]
    ZenLum = [round(i,3) for i in _df['zenith_illum']]

    if 'Wind Speed' in _df.keys():
        Wspeed = [round(i,3) for i in _df['Wind Speed']]
    for i in range(0, 8760):
        Tdew[i] = round(psychropy.psych(Patm[i], 'Tdb', Tdb[i], 'RH', RH[i] / 100, 'DP', 'SI'),1)
    _df['Dew Temperature (degC)']  =  Tdew
    return _df

def write_epw(save_path,file_name,df_):
    global GHRad, DNRad, DHRad, GHIll, DNIll, DHIll,  Wspeed, ExHorRad, Tdb, HorIR, ZenLum

    OPFILE = os.path.join(save_path, file_name+'.epw')
    ofile = open(OPFILE, "w", newline='')
    line1 = 'LOCATION,' + City + ',' + State + ',' + Country + ',customized weather file,' + str(wmosn) + ',' +\
            str(lat) + ',' + str(long) + ',' + str(tz) + ',' + str(elev) + '\n'
    ofile.write(line1)
    ofile.write(header[1])
    ofile.write(header[2])
    ofile.write(header[3])
    ofile.write(header[4])
    ofile.write('COMMENTS 1, ' + str(comm1) + '\n')
    # ofile.write(header[6])
    ofile.write('COMMENTS 2, TMY3 data downloaded Climate.Onebuilding.Org - Data merged with the Localized AMY File Creator (LA'+'F)\n')
    ofile.write('DATA PERIODS,1,1,Data,Sunday, 1/1,12/31\n')
    #
    writer = csv.writer(ofile, delimiter=',')
    Tdew = df_['Dew Temperature (degC)']
    Tdb =df_['Temperature']
    GHRad =  df_['GHI']
    DNRad = df_['DNI']
    DHRad = df_['DHI']
    if 'Wind Speed' in df_.keys():
        Wspeed =df_['Wind Speed']
    if 'DHI_extra' in df_.keys():
        ExHorRad = df_['DHI_extra']
    if 'horiz_ir' in df_.keys():
        HorIR = df_['horiz_ir']
    if 'GHI_illum' in df_.keys():
        GHIll = df_['GHI_illum']
    if 'DNIll' in df_.keys():
        DNIll = df_['DNI_illum']
    if 'DHI_illum' in df_.keys():
        DHIll = df_['DHI_illum']
    if 'zenith_illum' in df_.keys():
        ZenLum = df_['zenith_illum']
    for i in range(0, 8760):
        row = [int(Y[i]), int(M[i]), int(D[i]), int(HH[i]), int(MM[i]), DS,
               Tdb[i], Tdew[i], RH[i], Patm[i], ExHorRad[i], ExDirNormRad[i], HorIR[i],
               GHRad[i], DNRad[i], DHRad[i], GHIll[i], DNIll[i], DHIll[i], ZenLum[i],
               Wdir[i], Wspeed[i], TotSkyCover[i], OpSkyCover[i], Visib[i], CeilH[i],
               PresWeathObs[i], PresWeathCodes[i], PrecWater[i], AerOptDepth[i], SnowDepth[i],
               DSLS[i], Albedo[i], LiqPrecDepth[i], LiqPrecQuant[i]]
        writer.writerow(row)
    ofile.close()
    return 0
def create_epw_files(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    num_scenarios = int(editable_data['num_scenarios'])
    city=editable_data['city']
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    elev = float(editable_data['Altitude'])
    start_year = int(editable_data['starting_year'])
    end_year = int(editable_data['ending_year'])
    num_scenarios = int(editable_data['num_scenarios'])
    city = editable_data['city']
    state = editable_data['State']
    list_tmys = []
    if 'TMY_name' in editable_data.keys():
        TMY_name = editable_data['TMY_name']
        list_tmys.append(TMY_name)
    if 'TMY2_name' in editable_data.keys():
        TMY2_name = editable_data['TMY2_name']
        list_tmys.append(TMY2_name)
    if 'TMY3_name' in editable_data.keys():
        TMY3_name = editable_data['TMY3_name']
        list_tmys.append(TMY3_name)

    epw_file_name_TMY3= os.path.join(os.path.join(os.path.join(path_test,'Weather files'),'TMYs'),list_tmys[-1]+'.epw')
    scenarios_path = os.path.join(path_test,'ScenarioGeneration')
    folder_path = os.path.join(path_test,city)
    scenario_genrated = {}
    for scenario in range(num_scenarios):
        print('scenario', scenario)
        save_path = os.path.join(path_test,'ScenarioGeneration')
        scenario_genrated['SCN_'+str(scenario)]=pd.read_csv(os.path.join(save_path,'new_dependent_'+str(scenario)+'.csv'))
    read_tmy3(epw_file_name_TMY3)
    for scenario in range(num_scenarios):
        df=read_csv(scenario_genrated['SCN_'+str(scenario)])
        write_epw(scenarios_path,'SCN_'+str(scenario),df)

    dict_weather_csv = {}
    weather_data = {}
    for year in range(start_year,end_year+1):
        weather_data[year] = city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)
        dict_weather_csv[year]=  pd.read_csv(os.path.join(folder_path,weather_data[year]+'.csv'))
        #print(min(dict_weather_csv[year]['Temperature']))



    df_scenario_generated={}
    for i_temp in range(num_scenarios):
        for i_solar in range(num_scenarios):
                df_scenario_generated['T_'+str(i_temp)+'_S_'+str(i_solar)]=pd.read_csv(os.path.join(scenarios_path,'T_'+str(i_temp)+'_S_'+str(i_solar)+'.csv'))
    read_tmy3(epw_file_name_TMY3)
    for year in range(start_year,end_year+1):
        df=read_csv(dict_weather_csv[year])
        write_epw(scenarios_path,weather_data[year],df)


    read_tmy3(epw_file_name_TMY3)
    for i_temp in range(num_scenarios):
        for i_solar in range(num_scenarios):
            df=read_csv(df_scenario_generated['T_'+str(i_temp)+'_S_'+str(i_solar)])
            write_epw(scenarios_path,'T_'+str(i_temp)+'_S_'+str(i_solar),df)
