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
from DES_weather_analysis import weather_file_PDFs, psychropy
from DES_weather_analysis import solar_irradiance
from DES_weather_analysis.solar_irradiance import aoi, get_total_irradiance
from DES_weather_analysis.solar_position import get_solarposition
from DES_weather_analysis.irradiance import get_extra_radiation, clearness_index, _disc_kn,OrderedDict
from DES_weather_analysis.skymodel import estimate_illuminance_from_irradiance, calc_horizontal_infrared
from pvlib import atmosphere, solarposition, tools

## Adopted from PVlib (https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/irradiance.html)
def disc(ghi, solar_zenith, datetime_or_doy, pressure=101325,min_cos_zenith=0.065, max_zenith=87, max_airmass=12):
    """
    Estimate Direct Normal Irradiance from Global Horizontal Irradiance
    using the DISC model.

    The DISC algorithm converts global horizontal irradiance to direct
    normal irradiance through empirical relationships between the global
    and direct clearness indices.

    The pvlib implementation limits the clearness index to 1.

    The original report describing the DISC model [1]_ uses the
    relative airmass rather than the absolute (pressure-corrected)
    airmass. However, the NREL implementation of the DISC model [2]_
    uses absolute airmass. PVLib Matlab also uses the absolute airmass.
    pvlib python defaults to absolute airmass, but the relative airmass
    can be used by supplying `pressure=None`.

    Parameters
    ----------
    ghi : numeric
        Global horizontal irradiance in W/m^2.

    solar_zenith : numeric
        True (not refraction-corrected) solar zenith angles in decimal
        degrees.

    datetime_or_doy : int, float, array, pd.DatetimeIndex
        Day of year or array of days of year e.g.
        pd.DatetimeIndex.dayofyear, or pd.DatetimeIndex.

    pressure : None or numeric, default 101325
        Site pressure in Pascal. If None, relative airmass is used
        instead of absolute (pressure-corrected) airmass.

    min_cos_zenith : numeric, default 0.065
        Minimum value of cos(zenith) to allow when calculating global
        clearness index `kt`. Equivalent to zenith = 86.273 degrees.

    max_zenith : numeric, default 87
        Maximum value of zenith to allow in DNI calculation. DNI will be
        set to 0 for times with zenith values greater than `max_zenith`.

    max_airmass : numeric, default 12
        Maximum value of the airmass to allow in Kn calculation.
        Default value (12) comes from range over which Kn was fit
        to airmass in the original paper.

    Returns
    -------
    output : OrderedDict or DataFrame
        Contains the following keys:

        * ``dni``: The modeled direct normal irradiance
          in W/m^2 provided by the
          Direct Insolation Simulation Code (DISC) model.
        * ``kt``: Ratio of global to extraterrestrial
          irradiance on a horizontal plane.
        * ``airmass``: Airmass

    References
    ----------
    .. [1] Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly
       Global Horizontal to Direct Normal Insolation", Technical
       Report No. SERI/TR-215-3087, Golden, CO: Solar Energy Research
       Institute, 1987.

    .. [2] Maxwell, E. "DISC Model", Excel Worksheet.
       https://www.nrel.gov/grid/solar-resource/disc.html

    See Also
    --------
    dirint
    """

    # this is the I0 calculation from the reference
    # SSC uses solar constant = 1367.0 (checked 2018 08 15)
    I0 = get_extra_radiation(datetime_or_doy, 1370., 'spencer')

    kt = clearness_index(ghi, solar_zenith, I0, min_cos_zenith=min_cos_zenith,
                         max_clearness_index=1)

    am = atmosphere.get_relative_airmass(solar_zenith, model='kasten1966')
    if pressure is not None:
        am = atmosphere.get_absolute_airmass(am, pressure)

    Kn, am = _disc_kn(kt, am, max_airmass=max_airmass)
    dni = Kn * I0

    bad_values = (solar_zenith > max_zenith) | (ghi < 0) | (dni < 0)
    dni = np.where(bad_values, 0, dni)

    output = OrderedDict()
    output['dni'] = dni
    output['kt'] = kt
    output['airmass'] = am
    output['dni_extra'] = I0

    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        output = pd.DataFrame(output, index=datetime_or_doy)

    return output

def erbs(ghi, zenith, datetime_or_doy, min_cos_zenith=0.065, max_zenith=87):
    r"""
    Estimate DNI and DHI from GHI using the Erbs model.

    The Erbs model [1]_ estimates the diffuse fraction DF from global
    horizontal irradiance through an empirical relationship between DF
    and the ratio of GHI to extraterrestrial irradiance, Kt. The
    function uses the diffuse fraction to compute DHI as

    .. math::

        DHI = DF \times GHI

    DNI is then estimated as

    .. math::

        DNI = (GHI - DHI)/\cos(Z)

    where Z is the zenith angle.

    Parameters
    ----------
    ghi: numeric
        Global horizontal irradiance in W/m^2.
    zenith: numeric
        True (not refraction-corrected) zenith angles in decimal degrees.
    datetime_or_doy : int, float, array, pd.DatetimeIndex
        Day of year or array of days of year e.g.
        pd.DatetimeIndex.dayofyear, or pd.DatetimeIndex.
    min_cos_zenith : numeric, default 0.065
        Minimum value of cos(zenith) to allow when calculating global
        clearness index `kt`. Equivalent to zenith = 86.273 degrees.
    max_zenith : numeric, default 87
        Maximum value of zenith to allow in DNI calculation. DNI will be
        set to 0 for times with zenith values greater than `max_zenith`.

    Returns
    -------
    data : OrderedDict or DataFrame
        Contains the following keys/columns:

            * ``dni``: the modeled direct normal irradiance in W/m^2.
            * ``dhi``: the modeled diffuse horizontal irradiance in
              W/m^2.
            * ``kt``: Ratio of global to extraterrestrial irradiance
              on a horizontal plane.

    References
    ----------
    .. [1] D. G. Erbs, S. A. Klein and J. A. Duffie, Estimation of the
       diffuse radiation fraction for hourly, daily and monthly-average
       global radiation, Solar Energy 28(4), pp 293-302, 1982. Eq. 1

    See also
    --------
    dirint
    disc
    """

    dni_extra = get_extra_radiation(datetime_or_doy)

    kt = clearness_index(ghi, zenith, dni_extra, min_cos_zenith=min_cos_zenith,
                         max_clearness_index=1)

    # For Kt <= 0.22, set the diffuse fraction
    df = 1 - 0.09*kt

    # For Kt > 0.22 and Kt <= 0.8, set the diffuse fraction
    df = np.where((kt > 0.22) & (kt <= 0.8),
                  0.9511 - 0.1604*kt + 4.388*kt**2 -
                  16.638*kt**3 + 12.336*kt**4,
                  df)

    # For Kt > 0.8, set the diffuse fraction
    df = np.where(kt > 0.8, 0.165, df)

    dhi = df * ghi

    dni = (ghi - dhi) / tools.cosd(zenith)
    bad_values = (zenith > max_zenith) | (ghi < 0) | (dni < 0)
    dni = np.where(bad_values, 0, dni)
    # ensure that closure relationship remains valid
    dhi = np.where(bad_values, ghi, dhi)

    data = OrderedDict()
    data['dni'] = dni
    data['dhi'] = dhi
    data['kt'] = kt
    #data['extra_DNI']= dni_extra

    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        data = pd.DataFrame(data, index=datetime_or_doy)

    return data


def scenario_analysis(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    list_tmys= []
    for i in range(5):
        if 'TMY'+str(i+1)+'_name' in editable_data.keys():
            TMY_name = editable_data['TMY'+str(i+1)+'_name']
            list_tmys.append(TMY_name)
    csv_file_name_TMY3= pd.read_csv(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),'TMYs'),list_tmys[-1]+'.csv'))
    weather_keys = ['GHI','DNI','DHI']
    key_temp = 'Temperature'
    RH = csv_file_name_TMY3['relative_humidity']
    Patm = csv_file_name_TMY3['atmospheric_pressure']
    num_scenarios = int(editable_data['num_scenarios'])
    scenario_genrated = {}
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    altitude = float(editable_data['Altitude']) #SLC altitude m
    surf_tilt = float(editable_data['solar_tilt']) #panels tilt degree
    surf_azimuth = float(editable_data['solar_azimuth']) #panels azimuth degree
    end_year = int(editable_data['ending_year'])
    for scenario in range(num_scenarios):
        print('scenario', scenario)
        save_path = os.path.join(path_test,'ScenarioGeneration')
        scenario_genrated['SCN_'+str(scenario)]=pd.read_csv(os.path.join(save_path,'dependent_'+str(scenario)+'.csv'))
        for key in weather_keys:
            for j in range(50):
                for i in range(8760):
                    if scenario_genrated['SCN_'+str(scenario)]['GHI'][i]!=0 and scenario_genrated['SCN_'+str(scenario)]['GHI'][i]<0:
                        scenario_genrated['SCN_'+str(scenario)]['GHI'][i]=(scenario_genrated['SCN_'+str(scenario)]['GHI'][i-1]+scenario_genrated['SCN_'+str(scenario)]['GHI'][i+1])/2
                    if scenario_genrated['SCN_'+str(scenario)]['Temperature'][i]<-70 or scenario_genrated['SCN_'+str(scenario)]['Temperature'][i]>60:
                        scenario_genrated['SCN_'+str(scenario)]['Temperature'][i]=(scenario_genrated['SCN_'+str(scenario)]['Temperature'][i-1]+scenario_genrated['SCN_'+str(scenario)]['Temperature'][i+1])/2
        GHI = scenario_genrated['SCN_'+str(scenario)]['GHI']
        Temp= scenario_genrated['SCN_'+str(scenario)]['Temperature']
        dti = pd.date_range(str(end_year)+"-01-01", periods=8760, freq="H", tz='US/Mountain')
        #print(dti)
        solar_position = get_solarposition(dti, lat, lon, altitude, pressure=None, method='nrel_numpy', temperature=12)
        solar_zenith = solar_position['zenith']
        solar_elevation = solar_position['elevation']
        #print(max(solar_zenith),solar_zenith)
        solar_azimuth =  solar_position['azimuth']
        poa_components_vector = []
        poa_global = []
        DNI = []
        DHI = []
        DNI_extra = []
        DHI_extra = []
        kt = []
        GHI_illum = []
        DNI_illum = []
        DHI_illum = []
        zenith_illum = []
        horiz_ir = []
        Tdew = []
        #print(len(GHI), len (solar_zenith), len(dti))
        for i in range(len(solar_zenith)):
            #print(disc(GHI[i], solar_zenith[i], dti[i]))
            #DNI.append(disc(GHI[i], solar_zenith[i], dti[i]))
            #print(i,solar_elevation[i],solar_zenith[i], GHI[i],dti[i])
            #print( solar_zenith[i],int(GHI[i]), float(erbs(GHI[i], solar_zenith[i], dti[i])['dhi']))
            DNI.append(erbs(GHI[i], solar_zenith[i], dti[i])['dni'])
            DHI.append(erbs(GHI[i], solar_zenith[i], dti[i])['dhi'])
            kt.append(erbs(GHI[i], solar_zenith[i], dti[i])['kt'])
            DNI_extra.append(disc(GHI[i], solar_zenith[i], dti[i])['dni_extra'])
            if GHI[i]<=0:
                dhi_extra = 0
            else:
                dhi_extra = GHI[i]/kt[i]
            DHI_extra.append(dhi_extra)
            #print(altitude, GHI[i], DNI[i], DHI[i])
            #print(csv_file_name_TMY3.keys())
            #print(solar_zenith[i])
            Tdew.append(round(psychropy.psych(Patm[i], 'Tdb', Temp[i], 'RH', RH[i] / 100, 'DP', 'SI'),1))
            poa_components_vector.append(get_total_irradiance(surf_tilt, surf_azimuth,
                                     solar_zenith[i], solar_azimuth[i],
                                    float(DNI[i]), float(GHI[i]), float(DHI[i]), dni_extra=None, airmass=None,
                                     albedo=.25, surface_type=None,
                                     model='isotropic',
                                     model_perez='allsitescomposite1990'))
            poa_global.append(poa_components_vector[i]['poa_global'])

            #if solar_zenith[i]<90:
            #    solar_elevation[i] = 90- solar_zenith[i]

            #else:
                #solar_zenith[i]  = solar_zenith[i] -180
                #solar_elevation[i] = 90- solar_zenith[i]
            illuminance_estimation = estimate_illuminance_from_irradiance(solar_elevation[i], GHI[i], DNI[i], DHI[i], float(csv_file_name_TMY3['temp_dew'][i]))
            #print(illuminance_estimation)
            GHI_illum.append(illuminance_estimation[0])
            DNI_illum.append(illuminance_estimation[1])
            DHI_illum.append(illuminance_estimation[2])
            if illuminance_estimation[3]<0:
                illuminance_inf=0
            else:
                illuminance_inf =  illuminance_estimation[3]
            zenith_illum.append(illuminance_inf)

            horiz_ir.append(calc_horizontal_infrared(float(csv_file_name_TMY3['total_sky_cover'][i]), Temp[i], float(csv_file_name_TMY3['temp_dew'][i])))

        scenario_genrated_final={}
        scenario_genrated_final['Temperature'] = Temp
        scenario_genrated_final['GTI'] = poa_global
        scenario_genrated_final['GHI'] = GHI
        scenario_genrated_final['DHI'] = DHI
        scenario_genrated_final['DNI'] = DNI
        scenario_genrated_final['DHI_extra'] = DHI_extra
        scenario_genrated_final['DNI_extra'] = DNI_extra
        scenario_genrated_final['kt'] = kt
        scenario_genrated_final['GHI_illum'] = GHI_illum
        scenario_genrated_final['DNI_illum'] = DNI_illum
        scenario_genrated_final['DHI_illum'] = DHI_illum
        scenario_genrated_final['zenith_illum'] = zenith_illum
        scenario_genrated_final['horiz_ir'] = horiz_ir
        df_scenario_generated=pd.DataFrame(scenario_genrated_final)
        df_scenario_generated.to_csv(os.path.join(save_path,'new_dependent_'+str(scenario)+'.csv'), index=False)
