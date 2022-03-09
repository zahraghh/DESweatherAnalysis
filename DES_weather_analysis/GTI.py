#import multi_operation_planning
#from multi_operation_planning.solar_irradiance import aoi, get_total_irradiance
#from multi_operation_planning.solar_position import get_solarposition
import csv
from csv import writer, reader
import pandas as pd
import datetime
import os
import sys
import DES_weather_analysis
from DES_weather_analysis import solar_irradiance
from DES_weather_analysis.solar_irradiance import aoi, get_total_irradiance
from DES_weather_analysis.solar_position import get_solarposition
from pvlib import atmosphere, solarposition, tools
class GTI_class:
    def __init__(self,year,path_test,weather_path,TMYs=None,AMYs=None):
        if TMYs is None:
            self.TMYs = None
        else:
            self.TMYs = TMYs
        if AMYs is None:
            self.AMYs = None
        else:
            self.AMYs = AMYs
        editable_data_path =os.path.join(path_test, 'editable_values.csv')
        editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
        self.weather_path = weather_path
        self.lat = float(editable_data['Latitude'])
        self.lon = float(editable_data['Longitude'])
        self.altitude = float(editable_data['Altitude']) #SLC altitude m
        self.surf_tilt = float(editable_data['solar_tilt']) #panels tilt degree
        self.surf_azimuth = float(editable_data['solar_azimuth']) #panels azimuth degree
        self.year = year
        if AMYs is None:
            self.weather_data = pd.read_csv(self.weather_path).reset_index().drop('index', axis = 1)
        else:
            self.weather_data = pd.read_csv(self.weather_path)
            self.weather_data = self.weather_data.rename(columns=self.weather_data.iloc[1]).drop([0,1], axis = 0).reset_index()

    def process_gti(self):
        if self.AMYs is None:
            if self.TMYs is None:
                DNI= self.weather_data['dni']
                DHI = self.weather_data['dhi']
                GHI = self.weather_data['ghi']
                dti = pd.date_range(str(self.year)+'-01-01', periods=len(GHI), freq='H')
            else:
                DNI= self.weather_data['dni']
                DHI = self.weather_data['dhi']
                GHI = self.weather_data['ghi']
            df = pd.DataFrame({'year': self.weather_data['year'],
                    'month': self.weather_data['month'],
                    'day': self.weather_data['day'],
                    'hour': self.weather_data['hour']-1})
        else:
            DNI= self.weather_data['DNI']
            DHI = self.weather_data['DHI']
            GHI = self.weather_data['GHI']
            df = pd.DataFrame({'year': pd.to_numeric(self.weather_data['Year']),
                       'month': pd.to_numeric(self.weather_data['Month']),
                       'day': pd.to_numeric(self.weather_data['Day']),
                       'hour': pd.to_numeric(self.weather_data['Hour'])})
        dti =  df.apply(lambda row: datetime.datetime(row.year, row.month, row.day, row.hour), axis=1)
        solar_position = get_solarposition(dti, self.lat, self.lon, self.altitude, pressure=None, method='nrel_numpy', temperature=12)
        solar_zenith = solar_position['zenith']
        solar_azimuth =  solar_position['azimuth']
        poa_components_vector = []
        poa_global = []
        for i in range(len(solar_zenith)):
            poa_components_vector.append(get_total_irradiance(self.surf_tilt, self.surf_azimuth,
                                     solar_zenith[i], solar_azimuth[i],
                                    float(DNI[i]), float(GHI[i]), float(DHI[i]), dni_extra=None, airmass=None,
                                     albedo=.25, surface_type=None,
                                     model='isotropic',
                                     model_perez='allsitescomposite1990'))
            poa_global.append(poa_components_vector[i]['poa_global'])
        if self.AMYs is None:
            csv_input = pd.read_csv(self.weather_path).reset_index().drop('index', axis = 1)
        else:
            self.weather_data = pd.read_csv(self.weather_path)
            csv_input = self.weather_data.rename(columns=self.weather_data.iloc[1]).drop([0,1], axis = 0).reset_index()
        csv_input['gti'] = poa_global
        #print(csv_input)
        #print(self.weather_path)
        csv_input.to_csv(self.weather_path, index=False)
        return poa_global
def GTI_results(year,path_test,folder_path,TMYs=None,AMYs=None):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    if AMYs is None:
        if TMYs is None:
            weather_file = GTI_class(year,path_test,folder_path)
            weather_file.process_gti()
        else:
            weather_file = GTI_class(year=year,path_test=path_test,weather_path=folder_path,TMYs=TMYs)
            weather_file.process_gti()
    else:
        weather_file = GTI_class(year=year,path_test=path_test,weather_path=folder_path,AMYs=AMYs)
        weather_file.process_gti()
