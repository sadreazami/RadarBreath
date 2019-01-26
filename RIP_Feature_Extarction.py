

import csv
import os
import cProfile
import time
import numpy as np
import pandas
import math
import warnings
import random
import pandas as pd
import matplotlib.pyplot as plt
#import pyemd 
import matplotlib.patches as mpatches
import scipy 



import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def chunks(l, k):
  for i in range(0, len(l), k):
    yield l[i:i+k]
    
def shortTermEnergy(frame):
  return sum( [ abs(x)**2 for x in frame ] ) / len(frame)

def entropyOfEnergy(frame, numSubFrames):
  lenSubFrame = int(np.floor(len(frame) / numSubFrames))
  shortFrames = list(chunks(frame, lenSubFrame))
  energy      = [ shortTermEnergy(s) for s in shortFrames ]
  totalEnergy = sum(energy)
  energy      = [ e / totalEnergy for e in energy ]

  entropy = 0.0
  for e in energy:
    if e != 0:
      entropy = entropy - e * np.log2(e)

  return entropy

def ReadData(filename):
    df = pandas.read_csv(filename, sep=',', na_filter=True)
    data = df.values
    return data
    
cls()
plt.close('all')
dic = {'SUB': ['S02_','S03_','S04_'], 'POS':['SIT_'],'Rep':['R1_','R2_','R3_']}
mycolor ='gkr'
DownSampleRate = 1
fs = 1000/DownSampleRate
RPS_Tau = 1000/DownSampleRate
RPS_Dim = 2
Segment_length = fs*5 # 10 seconds of signal
Subplot_Num = int(900000/Segment_length)
print(dic['SUB'][0][1])


Signal_ALL =np.empty((0,Segment_length), float64)

y_pred = np.empty((0,1), int)
PSD_ALL= np.empty((0,1), float64)
Entropy_ALL = np.empty((0,1), float64)
for subject in dic['SUB']:
    for posture in dic['POS']:
        for repitition in dic['Rep']:
            filename = 'CSVFiles/'+subject+posture+repitition+'ECGRIP.csv'
            print(filename)
            x = ReadData(filename)
            t = x[:,1]
            Signal = x[:,2]
            if len(Signal)<900000:
                Signal = np.append(Signal[range(900000-len(Signal))],Signal)
                
            extra_len = len(Signal)%Segment_length
            Signal = np.delete(Signal, range(extra_len))
            Num_Segment = int(len(Signal)/Segment_length)
            #Signal = scipy.signal.decimate(Signal,DownSampleRate )
            
            
            for segment_index in range( Num_Segment ):
               Segment = Signal[range(Segment_length)] 
               Entropy = entropyOfEnergy(Segment, 50)
               Entropy_ALL = np.append(Entropy_ALL,Entropy)
               #Segment = (Segment -np.mean(Segment))/np.std(Segment)
               Signal = np.delete(Signal, range(Segment_length))
               Signal_ALL =np.vstack([Signal_ALL, Segment])
               # Estimate PSD `S_xx_welch` at discrete frequencies `f_welch`
               f_welch, S_xx_welch = scipy.signal.welch(Segment, fs=fs,nperseg = 2048)
                # Integrate PSD over spectral bandwidth
                # to obtain signal power `P_welch`
               df_welch = f_welch[1] - f_welch[0]
               PSD = np.sum(S_xx_welch[range(2)])/np.sum(S_xx_welch[range(10)])#* df_welc* df_welc#h
               PSD_ALL = np.append(PSD_ALL,PSD)
          