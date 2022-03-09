from collections import defaultdict
import os
import sys
import pandas as pd
import diyepw
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import json
import matplotlib.pyplot as plt
from copulas.datasets import sample_bivariate_age_income
from copulas.multivariate import GaussianMultivariate
import DES_weather_analysis
from  DES_weather_analysis import EPW_to_csv,GTI
def weather_PDS_data(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    num_scenarios = int(editable_data['num_scenarios'])
    start_year = 1998
    end_year = 2019
    city=editable_data['city']
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    folder_path = os.path.join(path_test,str(city))
    dict_weather_csv = {}
    dict_EPWs = {}
    list_years = []
    list_tmys =[]
    list_fmys = []
    for year in range(start_year,end_year+1):
        weather_data = city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)
        list_years.append(weather_data)
    if 'TMY_name' in editable_data.keys():
        TMY_name = editable_data['TMY_name']
        list_tmys.append(TMY_name)
    if 'TMY2_name' in editable_data.keys():
        TMY2_name = editable_data['TMY2_name']
        list_tmys.append(TMY2_name)
    if 'TMY3_name' in editable_data.keys():
        TMY3_name = editable_data['TMY3_name']
        list_tmys.append(TMY3_name)
    if 'FMY1_name' in editable_data.keys():
        FMY1_name = editable_data['FMY1_name']
        list_fmys.append(FMY1_name)
    if 'FMY2_name' in editable_data.keys():
        FMY2_name = editable_data['FMY2_name']
        list_fmys.append(FMY2_name)
    dict_EPWs['AMYs']=list_years
    dict_EPWs['FMYs']=list_fmys
    dict_EPWs['TMYs']=list_tmys
    def convert_to_csv(dict_EPWs):
        for key in dict_EPWs.keys():
            for epw_file_name in dict_EPWs[key]:
                if key!= 'AMYs':
                    if key =='FMYs':
                        data, meta = EPW_to_csv.read_epw(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.epw'),FMYs='yes')
                    elif key =='TMY':
                        data, meta = EPW_to_csv.read_epw(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.epw'))
                    data.to_csv(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.csv'), index = False, header=True)
                    dict_weather_csv[epw_file_name] =  pd.read_csv(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.csv'))[0:8760]
                else:
                    dict_weather_csv[epw_file_name] =  pd.read_csv(os.path.join(folder_path,epw_file_name+'.csv'))[0:8760]
        return dict_weather_csv
    #Only reading the weather files
    temp_air = []
    gti = []
    ghi = []
    dni = []
    dhi = []
    wind_speed = []

    for key in dict_EPWs.keys():
        for epw_file_name in dict_EPWs[key]:
            if key=='AMYs':
                dict_weather_csv[epw_file_name] =   pd.read_csv(os.path.join(folder_path,epw_file_name+'.csv'))[0:8760]
                weather_path = os.path.join(folder_path,epw_file_name+'.csv')
                if 'gti' not in dict_weather_csv[epw_file_name].keys():
                    year = int(epw_file_name[-4:])
                    dict_weather_csv[epw_file_name]['gti'] = GTI.GTI_results(year=year,path_test=path_test, folder_path=weather_path, AMYs='AMYs')
                    dict_weather_csv[epw_file_name] = pd.read_csv(os.path.join(folder_path,epw_file_name+'.csv'))[0:8760]
                temp_air.append(dict_weather_csv[epw_file_name]['Temperature'])
                ghi.append(dict_weather_csv[epw_file_name]['GHI'])
                dni.append(dict_weather_csv[epw_file_name]['DNI'])
                dhi.append(dict_weather_csv[epw_file_name]['DHI'])
                wind_speed.append(dict_weather_csv[epw_file_name]['Wind Speed'])
            else:
                dict_weather_csv[epw_file_name] =  pd.read_csv(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.csv'))[0:8760]
                weather_path = os.path.join(os.path.join(os.path.join(path_test,'Weather files'),key),epw_file_name+'.csv')
                if  key=='FMYs':
                    if 'gti' not in dict_weather_csv[epw_file_name].keys():
                        year = int(epw_file_name[-4:])
                        dict_weather_csv[epw_file_name]['gti'] = GTI.GTI_results(year,path_test,weather_path)
                elif key=='TMYs':
                    if 'gti' not in dict_weather_csv[epw_file_name].keys():
                        year=0
                        dict_weather_csv[epw_file_name]['gti'] = GTI.GTI_results(year=year,path_test=path_test,weather_path=weather_path,TMYs='TMYs')
                temp_air.append(dict_weather_csv[epw_file_name]['temp_air'])
                wind_speed.append(dict_weather_csv[epw_file_name]['wind_speed'])
                ghi.append(dict_weather_csv[epw_file_name]['ghi'])
                dni.append(dict_weather_csv[epw_file_name]['dni'])
                dhi.append(dict_weather_csv[epw_file_name]['dhi'])
            gti.append(dict_weather_csv[epw_file_name]['gti'])

    #weather_params={'ghi':ghi, 'dni':dni,'dhi':dhi,'temp_air':temp_air, 'wind_speed':wind_speed}
    weather_params={'gti': gti,'ghi':ghi, 'dni':dni,'dhi':dhi,'temp_air':temp_air}

    # Create models from data
    def best_fit_distribution(data,  ax=None):
      """Model data by finding best fit distribution to data"""
      # Get histogram of original data
      y, x = np.histogram(data, bins='auto', density=True)
      x = (x + np.roll(x, -1))[:-1] / 2.0
      # Distributions to check
      #DISTRIBUTIONS = [
        #st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,st.hypsecant,
        #st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        #st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        #st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        #st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.invgamma,st.invgauss,
        #st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,
        #st.levy,st.levy_l,st.levy_stable,  #what's wrong with these distributions?
        #st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        #st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        #st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        #st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
      #DISTRIBUTIONS = [st.norm, st.uniform, st.expon, st.beta]
      DISTRIBUTIONS = [st.beta,st.norm, st.uniform, st.expon,st.weibull_min,st.weibull_max,st.gamma,st.chi,st.lognorm,st.cauchy,st.triang,st.f]

      # Best holders
      best_distribution = st.norm  # random variables
      best_params = (0.0, 1.0)
      best_sse = np.inf
      # Estimate distribution parameters from data
      for distribution in DISTRIBUTIONS:
          # fit dist to data
          params = distribution.fit(data)

          warnings.filterwarnings("ignore")
          # Separate parts of parameters
          arg = params[:-2]
          loc = params[-2]
          scale = params[-1]
          # Calculate fitted PDF and error with fit in distribution
          pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
          sse = np.sum(np.power(y - pdf, 2.0))
          # if axis pass in add to plot
          try:
              if ax:
                  pd.Series(pdf, x).plot(ax=ax)
              end
          except Exception:
              pass
          # identify if this distribution is better
          if best_sse > sse > 0:
              best_distribution = distribution
              best_params = params
              best_sse = sse
      return (best_distribution.name, best_params)
    def fit_and_plot(dist,data,min_data, max_data):
        params = dist.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        y, x = np.histogram(data, bins='auto', density=True)
        bin_centers = 0.5*(x[1:] + x[:-1])
        fig_monte=plt.figure('probability distribution', figsize=(8,5), dpi=100)
        nplt, binsplt, patchesplt = plt.hist(data, bins='auto', range=(min_data,max_data), density= True)
        pdf= dist.pdf(bin_centers, loc=loc, scale=scale, *arg)
        percent_formatter = partial(to_percent ,k = sum(nplt))
        formatter = FuncFormatter(percent_formatter)    # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        ax = fig_monte.add_subplot(111)
        ax.plot(bin_centers, pdf, linestyle ='-', color = "m", linewidth =3)
        ax.set_xlabel('DNI')
        ax.set_ylabel('Probability of DNI')
        plt.show()
        return dist, pdf
    def to_percent(nplt,position, k):
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        s = str(round(100*nplt/k,0))
        # The percent symbol needs escaping in latex
        if matplotlib.rcParams['text.usetex'] is True:
            return s + r'$\%$'
        else:
            return s + '%'
    data_hourly = []
    data_daily = []
    def data_classification(name):
        list_weather = defaultdict(list)
        for index in range(8760):
            for year in range(len(weather_params[name])):
                list_weather[index].append(float(weather_params[name][year][index]))
        list_weather = dict(list_weather)
        with open(os.path.join(path_test, 'UA_'+name+'.json'), 'w') as fp:
            json.dump(list_weather,fp)
        return list_weather

    # Find best fit distribution
    def probability_distribution(name,weather_data,path_test):
        best_fit_input = defaultdict(list)
        df_object = {}
        df_object_all  = pd.DataFrame(columns = ['Index in year','Best fit','Best loc','Best scale','Mean','STD'])
        gen_scenarios_param = []
        for index in range(8760):
            if np.std(weather_data[index])==0:
                #print('here',index)
                best_fit_name = 'constant'
                best_fit_params= np.mean(data)
            else:
                best_fit_name, best_fit_params = best_fit_distribution(weather_data[index])
            gen_scenarios_param.append([best_fit_name, best_fit_params])
        with open(os.path.join(path_test, 'best_fit_'+name+'.json'), 'w') as fp:
            json.dump(gen_scenarios_param,fp)
    for name in weather_params.keys():
        try:
            f=open(os.path.join(path_test, 'UA_'+name+'.json'))
        except IOError:
            weather_params_data= data_classification(name)
            try:
                f=open(os.path.join(path_test, 'best_fit_'+name+'.json'))
            except IOError:
                probability_distribution(name,weather_params_data,path_test)
