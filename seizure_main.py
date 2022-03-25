
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import scipy.io
from matplotlib import pyplot as plt

import mne
from scipy.io import loadmat
import os,time,pandas as pd,scipy,numpy as np
from scipy import signal,math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import antropy
import yasa,mne,emd
from antropy import fractal
from prettytable import PrettyTable
from mne.preprocessing import ICA
from scipy import stats
import seaborn as sns
from scipy.signal import hilbert
import tsfel
import pandas as pd
from PIL import Image
bands = [(0.1, 4, 'delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 70, 'Low Gamma'), (70, 180, 'High Gamma')]
import requests



# Fxn
@st.cache

def load_image(image_file):
	img = Image.open(image_file)
	return img 
def spectogram(x,sampling_frequency):
    '''
    Generate Time-Frequency image of signal
    
    '''
    
    f, t, Sxx = signal.spectrogram(x, sampling_frequency)
    plt.pcolormesh(t, f, np.log10(np.abs(Sxx)), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
   
    

    
def extract_data_from_mat(file_path):
    '''
    Extract data from .mat file
    
    '''
    
    [(struct_name, shape, dtype)] = scipy.io.whosmat(file_path)
    f=loadmat(file_path, struct_as_record=False, squeeze_me=True)[struct_name]
            
    # extract data from .mat file 
    data=f.data.astype("float64")
    data_len_sec=f.data_length_sec
    sampling_frequency=f.sampling_frequency
    channels=f.channels.tolist()
    num_electrodes=len(channels)
    
    
    return data,data_len_sec,sampling_frequency,channels,num_electrodes
def covert_mat_mne_format(data,data_len_sec,sampling_frequency,channels,num_electrodes):
    ch_names=channels
    sfreq=sampling_frequency
    ch_types = ['eeg'] * len(channels)
    
    info=mne.create_info(ch_names, ch_types=ch_types,sfreq=sfreq)
    info['description'] = 'Patient preictal Epilepsy'
    
    raw=mne.io.RawArray(data,info,verbose=False)
    return raw
    


def preprocess_file(file_path):
    
    
    data,data_len_sec,sampling_frequency,channels,num_electrodes=extract_data_from_mat(file_path)
    raw=covert_mat_mne_format(data,data_len_sec,sampling_frequency,channels,num_electrodes)
   
    # coverting multiple channels to surrogate channel using average filtering
    data1 = raw.get_data()
    #raw.plot_psd(area_mode='range', tmax=10.0,  average=False)
    sugg_chann=np.mean(data1,axis=0)
    st.write("Spectrogram of surrogate signal")
    
    col1, col2 = st.columns(2)
   
    
    with col1:
        col1.header("Before  Power line Interface")
        bfr=spectogram(sugg_chann,sampling_frequency)
        st.pyplot(bfr)   
    raw1=raw.notch_filter(60,  filter_length='auto', phase='zero')
    sugg_chann=np.mean(raw1.get_data(),axis=0)  
          
    with col2:
       col2.header("After  Power line Interface")
       spec=spectogram(sugg_chann,sampling_frequency)  
       st.pyplot(spec)
    
    st.write("Power spectral density plot of surrogate signal")
    
    raww=covert_mat_mne_format(data,data_len_sec,sampling_frequency,channels,num_electrodes)
    col11, col12 = st.columns(2)
    with col11:
       col11.header("PSD before Powerline noise")
       psd_plt=raww.plot_psd(area_mode='range',fmax=150 ,tmax=10.0,  average=False)
       plt.show()
       st.pyplot()
       
    raw1=raww.notch_filter(60,  filter_length='auto', phase='zero')
           
    with col12:  
       col2.header("PSD after Powerline noise")
       psd_plt=raww.plot_psd(area_mode='range',fmax=150 ,tmax=10.0,  average=False)
       plt.show()
       st.pyplot() 
         
    sugg_chann=np.mean(raww.get_data(),axis=0)  
    
       
    # Denoising surrogate channel using Empirical mode decomposition
    IMFs=emd.sift.sift(sugg_chann)
     
    st.write("IMFs plots of Surrogate channel")
    abc=emd.plotting.plot_imfs(IMFs, scale_y=True, cmap=True)
    st.pyplot(abc)   
    
    filtr_IMFs1=IMFs.T[1]+IMFs.T[2]+IMFs.T[3]+IMFs.T[4]
    filtr_IMFs_last=IMFs.T[-1]+IMFs.T[-2]+IMFs.T[-3]+IMFs.T[-4]
       
    
    
    return filtr_IMFs1,filtr_IMFs_last,sampling_frequency


def feature_extraction(filtr_IMFs,sampling_frequency):
    df=pd.DataFrame()
    mn,vari,skewn,kurt,abs_energy,pk_pk_dis=[],[],[],[],[],[]
    spec_centroid,spec_kurt,spec_skewness,spec_variation,ttl_energy=[],[],[],[],[]
    
    # Statistical Features
    mn.append(tsfel.feature_extraction.features.calc_mean(filtr_IMFs))
    vari.append(tsfel.feature_extraction.features.calc_var(filtr_IMFs))
    skewn.append(tsfel.feature_extraction.features.skewness(filtr_IMFs))
    kurt.append(tsfel.feature_extraction.features.kurtosis(filtr_IMFs))
    abs_energy.append(tsfel.feature_extraction.features.abs_energy(filtr_IMFs))
    pk_pk_dis.append(tsfel.feature_extraction.features.pk_pk_distance(filtr_IMFs))
            
    # Spectral Features
    spec_centroid.append(tsfel.feature_extraction.features.spectral_centroid(filtr_IMFs, sampling_frequency))
    spec_kurt.append(tsfel.feature_extraction.features.spectral_kurtosis(filtr_IMFs, sampling_frequency))
    spec_skewness.append(tsfel.feature_extraction.features.spectral_skewness(filtr_IMFs, sampling_frequency))
    spec_variation.append(tsfel.feature_extraction.features.spectral_variation(filtr_IMFs, sampling_frequency))
    ttl_energy.append(tsfel.feature_extraction.features.total_energy(filtr_IMFs, sampling_frequency))
          
    ## avg. power band
    df1 = yasa.bandpower(filtr_IMFs, sampling_frequency,bands=bands)
    df1.drop(['FreqRes', 'TotalAbsPow','Relative'], axis=1, inplace=True)
           
    df=df.append(df1)            
            
    df["Mean"]=mn
    df["variance"]=vari
    df["Skewness"]=skewn
    df["Kurtosis"]=kurt
    df["Absolute_energy"]=abs_energy
    df["peak_peak_distance"]=pk_pk_dis
    
    df["Spectral Centroid"]=spec_centroid
    df["Spectral_Kurtosis"]=spec_kurt
    df["spectral_skewness"]=spec_skewness
    df["Spectral_variation"]=spec_variation
    df["Total_Energy"]=ttl_energy
 
    
    
   
    return df
        








def main():
        st.set_page_config(
             page_title="Seizure Prediction",
             layout="wide",
             initial_sidebar_state="expanded"
        )
        st.set_option('deprecation.showPyplotGlobalUse', False)
        with st.sidebar:
            mat_file=st.file_uploader("Upload Image",type=['.mat'])
            if st.button("Upload"):
               if mat_file is not None:
                    file_details = {"Filename":mat_file.name,"FileType":mat_file.type,"FileSize":mat_file.size}
                    st.write(file_details)
               else:
                    st.write("Wrong file format")
                
              
            menu = ["Select Any ","Dog_1","Dog_2","Dog_3","Dog_4","DOg_5","Patient_1","Patient_2"]
            choice = st.sidebar.selectbox("Select Model",menu)     
             
              
        _CWD = os.getcwd() 
            
        if choice == "Patient_1":
               final_model_file = os.path.join(_CWD,'Patient_1_model.pkl')
               if not os.path.isfile(final_model_file): # If the model is not present
                   url = r'https://github.com/sadikaVER/Seizure_prediction_streamlit/blob/main/Patient_1_model.pkl?raw=true'
                   resp = requests.get(url)
                   with open(final_model_file, 'wb') as fopen:
                        fopen.write(resp.content)
               with open(final_model_file, 'rb') as file:
                     	load_model = joblib.load(file)
               datadf=pd.DataFrame()
               st.subheader("Patient_1")
               
                
               first_4,last_4,sampling_frequency=preprocess_file(mat_file)
               dataf=feature_extraction(first_4,sampling_frequency)
               
               
               
               
               X_test=dataf.to_numpy()
               
               grid_predictions = load_model.predict(X_test) 
               if int(grid_predictions)==int(1):
                    st.write("Prediction: Preictal")
               else:
                    st.write("Prediction : Interictal")
                
        elif choice == "Dog_1":
               dog_final_model_file = os.path.join(_CWD,'Dog_1_model.pkl')
               if not os.path.isfile(dog_final_model_file): # If the model is not present
                   url = r'https://github.com/sadikaVER/Seizure_prediction_streamlit/blob/main/Dog_1_model.pkl?raw=true'
                   resp = requests.get(url)
                   with open(dog_final_model_file, 'wb') as fopen:
                        fopen.write(resp.content)
               with open(dog_final_model_file, 'rb') as file:
                     	load_model = joblib.load(file)
               datadf=pd.DataFrame() 
              
               st.subheader("Dog_1")
               
                
               first_4,last_4,sampling_frequency=preprocess_file(mat_file)
               dataf=feature_extraction(first_4,sampling_frequency)
               
               
               
               
               X_test=dataf.to_numpy()
               
               grid_predictions = load_model.predict(X_test) 
               if int(grid_predictions)==int(1):
                    st.write("Prediction: Preictal")
               else:
                    st.write("Prediction : Interictal")            
        
        elif choice == "Patient_2":
		Patient_final_model_file = os.path.join(_CWD,'Patient_2_model.pkl')
		if not os.path.isfile(Patient_final_model_file): # If the model is not present
                   url = r'https://github.com/sadikaVER/Seizure_prediction_streamlit/blob/main/Patient_2_model.pkl?raw=true'
                   resp = requests.get(url)
                   with open(final_model_file, 'wb') as fopen:
                        fopen.write(resp.content)
                with open(final_model_file, 'rb') as file:
                     	load_model = joblib.load(file)
                datadf=pd.DataFrame()
                st.subheader("Patient_2")
                first_4,last_4,sampling_frequency=preprocess_file(mat_file)
                dataf=feature_extraction(first_4,sampling_frequency)
               
               
                load_model=joblib.load(model_file)
                X_test=dataf.to_numpy()
              
                grid_predictions = load_model.predict(X_test) 
                st.write(grid_predictions)
                if int(grid_predictions)==int(1):
                    st.write("Prediction: Preictal")
                else:
                    st.write("Prediction : Interictal")        
        elif choice == "Dog_2":
               dog_final_model_file = os.path.join(_CWD,'Dog_2_model.pkl')
               if not os.path.isfile(dog_final_model_file): # If the model is not present
                   url = r'https://github.com/sadikaVER/Seizure_prediction_streamlit/blob/main/Dog_2_model.pkl?raw=true'
                   resp = requests.get(url)
                   with open(dog_final_model_file, 'wb') as fopen:
                        fopen.write(resp.content)
               with open(dog_final_model_file, 'rb') as file:
                     	load_model = joblib.load(file)
               datadf=pd.DataFrame() 
              
               st.subheader("Dog_2")
               
                
               first_4,last_4,sampling_frequency=preprocess_file(mat_file)
               dataf=feature_extraction(first_4,sampling_frequency)
               
               
               
               
               X_test=dataf.to_numpy()
               
               grid_predictions = load_model.predict(X_test) 
               if int(grid_predictions)==int(1):
                    st.write("Prediction: Preictal")
               else:
                    st.write("Prediction : Interictal")            
        elif choice == "Dog_3":
               dog_final_model_file = os.path.join(_CWD,'Dog_3_model.pkl')
               if not os.path.isfile(dog_final_model_file): # If the model is not present
                   url = r'https://github.com/sadikaVER/Seizure_prediction_streamlit/blob/main/Dog_3_model.pkl?raw=true'
                   resp = requests.get(url)
                   with open(dog_final_model_file, 'wb') as fopen:
                        fopen.write(resp.content)
               with open(dog_final_model_file, 'rb') as file:
                     	load_model = joblib.load(file)
               datadf=pd.DataFrame() 
              
               st.subheader("Dog_3")
               
                
               first_4,last_4,sampling_frequency=preprocess_file(mat_file)
               dataf=feature_extraction(first_4,sampling_frequency)
               
               
               
               
               X_test=dataf.to_numpy()
               
               grid_predictions = load_model.predict(X_test) 
               if int(grid_predictions)==int(1):
                    st.write("Prediction: Preictal")
               else:
                    st.write("Prediction : Interictal")            
        elif choice == "Dog_4":
               dog_final_model_file = os.path.join(_CWD,'Dog_4_model.pkl')
               if not os.path.isfile(dog_final_model_file): # If the model is not present
                   url = r'https://github.com/sadikaVER/Seizure_prediction_streamlit/blob/main/Dog_4_model.pkl?raw=true'
                   resp = requests.get(url)
                   with open(dog_final_model_file, 'wb') as fopen:
                        fopen.write(resp.content)
               with open(dog_final_model_file, 'rb') as file:
                     	load_model = joblib.load(file)
               datadf=pd.DataFrame() 
              
               st.subheader("Dog_4")
               
                
               first_4,last_4,sampling_frequency=preprocess_file(mat_file)
               dataf=feature_extraction(first_4,sampling_frequency)
               
               
               
               
               X_test=dataf.to_numpy()
               
               grid_predictions = load_model.predict(X_test) 
               if int(grid_predictions)==int(1):
                    st.write("Prediction: Preictal")
               else:
                    st.write("Prediction : Interictal")            
        elif choice == "Dog_5":
               dog_final_model_file = os.path.join(_CWD,'Dog_5_model.pkl')
               if not os.path.isfile(dog_final_model_file): # If the model is not present
                   url = r'https://github.com/sadikaVER/Seizure_prediction_streamlit/blob/main/Dog_5_model.pkl?raw=true'
                   resp = requests.get(url)
                   with open(dog_final_model_file, 'wb') as fopen:
                        fopen.write(resp.content)
               with open(dog_final_model_file, 'rb') as file:
                     	load_model = joblib.load(file)
               datadf=pd.DataFrame() 
              
               st.subheader("Dog_5")
               
                
               first_4,last_4,sampling_frequency=preprocess_file(mat_file)
               dataf=feature_extraction(first_4,sampling_frequency)
               
               
               
               
               X_test=dataf.to_numpy()
               
               grid_predictions = load_model.predict(X_test) 
               if int(grid_predictions)==int(1):
                    st.write("Prediction: Preictal")
               else:
                    st.write("Prediction : Interictal")            
                                                                                     
        else:
           st.subheader("About")
           st.info("Built with Streamlit")
           st.info("Sadika Verma")
           st.text("sadika.verma@gmail.com")



if __name__ == '__main__':
	main()




