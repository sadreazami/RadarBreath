
import matplotlib.patches as mpatches
import itertools
import csv
import time
import numpy as np
import pandas as pd
import math
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal 
import seaborn as sns
import scipy.io as sio
from scipy import stats
import random
from sklearn import linear_model
import numpy
from scipy.signal import chirp, hilbert
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.fftpack import fft, ifft
from sklearn.metrics import mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
plt.close('all')


def calc_MI2(x, y, bins):
    c_xy = numpy.histogram2d(x, y, bins)[0]
    mi2 = mutual_info_score(None, None, contingency=c_xy)
    return mi2

def crosscor(x,y):
    s=numpy.corrcoef(x,y)[0, 1]
    s=numpy.abs(s)
    return s
#    
def autocorr_by_hand(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x)-lag)]
    y2 = x[lag:]
    # Subtract the subseries means
    sum_product = np.sum((y1-np.mean(y1))*(y2-np.mean(y2)))
    # Normalize with the subseries stds
    return sum_product / ((len(x) - lag) * np.std(y1) * np.std(y2))

def acf_by_hand(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x)-lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x to calculate Cov
    sum_product = np.sum((y1-np.mean(x))*(y2-np.mean(x)))
    # Normalize with var of whole series
    return sum_product / ((len(x) - lag) * np.var(x))
    
def AMI(x):
    data=x
    data=data.astype(numpy.float64)
    tau_max = 60
    mis = []
    mi=[]
    for tau in range(1, tau_max):
        unlagged = data[:-tau]
        lagged = numpy.roll(data, -tau)[:-tau]
        mis.append(calc_MI2(unlagged,lagged,50))
#        S=crosscor(unlagged, lagged)
        S=autocorr_by_hand(unlagged, lag=tau)
        mi.append(S)
        
    plt.figure()    
    plt.plot(range(len(mis)), mis, color='r')
    plt.figure()    
    plt.plot(range(len(mi)), mi, color='b')   
#    f2 = plt.figure()
#    f2, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=True)
#    ax1.plot(range(len(mis)),mis)
#    ax1.set_title('')
#    ax2.plot(range(len(mi)), mi, color='g')
#    f2.subplots_adjust(hspace=0)
#    plt.setp([a.get_xticklabels() for a in f2.axes[:-1]], visible=False)
#    
    return mis,mi
# ((x.max())/100)   
def addsway(x, f0, lam, dur):
    x_scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    x = (x_scaler.fit_transform(x))
    t = np.arccos(x)
    y = t*lam/(2*np.pi)
    d_ran = ((x.max())/1)*np.sin(2*np.pi*(np.linspace(0, 5, dur)))
#    d_ran = 0.001*(np.random.uniform(0, 1, x.shape))
    yy2=np.cos((2*np.pi*(y))/lam)
    out = (x_scaler.inverse_transform(yy2))
    return out
    
def genSine(f0, fs, dur):

    t = np.arange(dur)
    sinusoid = np.sin(2*np.pi*t*(f0/fs))
    return sinusoid

    
def genSine2(T, f0, f1,dur):
    t = np.linspace(0, T, dur, endpoint=False)
    y = chirp(t, f0, T, f1, method='logarithmic')
#    analytic_signal = hilbert(y)
#    yy = np.abs(analytic_signal)
    return y
    
def genSine3(f, fs,sample):
 
    x = np.arange(sample)
    noise=0.3 * np.random.normal(size=sample)
#    noise = 0.0008*np.asarray(random.sample(range(0,sample),sample))
    y2 = np.sin(2 * np.pi * f * x / fs)+noise
    return y2

def ReadData(filename):
    df = pd.read_csv(filename, sep=',', na_filter=True)
    data = df.values
    return data
    
def FindZone(x):
    ZoneSum = x.sum(axis=0)
    ZoneNum = list(zip(*ZoneSum.nonzero()))
    return ZoneNum[0][0]
    
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma  
   
dic = {'SUB': ['S02_','S03_','S04_'], 'POS':['SIT_'],'Rep':['R1_','R2_','R3_']}
  #',,'S04_'
#dic = {'SUB': ['S02_'], 'POS':['SIT_'],'Rep':['R3_']}
       
DownSampleRate = 10
fs = 1480/DownSampleRate
RPS_Tau = 0.05 #sec
RPS_Dim = 2
Segment_length = 5 # 10 seconds of signal
numberOfRows = 180
# create dataframe
Dataset = pd.DataFrame(columns=('Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum'))

for subject in dic['SUB']:
    for posture in dic['POS']:
        for repitition in dic['Rep']:
            filename = 'Radar_Data/'+subject+posture+repitition+'RADAR.csv'
            print(filename)
            x = ReadData(filename)
            ZoneNum = FindZone(x)
            Signal = x[:,ZoneNum]
           
#            Signal=Signal-np.mean(Signal)
#            plt.figure()
#            plt.plot(Signal,'g')
            Signal = scipy.signal.decimate(Signal,DownSampleRate )

#            Signal = addsway(Signal, 1.3, 0.0125, len(Signal))
#            Signal2 = Signal + 3.0*np.sin(w*(np.linspace(0, 15, len(Signal))))+9724 
#            Signal2 = Signal + 100*genSine3(1,150000,len(Signal))
#            oo=700*genSine2(30,1,1,len(Signal))
#            Signal2 =  Signal +oo 

#            plt.figure()
#            plt.plot(Signal,'g')
#            plt.plot(Signal2, 'b')
#            plt.plot(Signal2-Signal, 'y')

            if len(Signal)>=fs*numberOfRows*Segment_length:
               Signal = np.array(Signal[range(int(fs*numberOfRows*Segment_length))])
            else:
               Signal = np.concatenate((Signal,np.zeros(fs*numberOfRows*Segment_length-len(Signal))),axis = 0)
            #print(len(Signal)/fs)
            
#            Signal = Signal + 1*(np.random.uniform(min(Signal), 10*min(Signal)+1000, Signal.shape))
#            Signal2 = Signal + 3.0*np.sin(np.linspace(0, 20*np.pi, len(Signal))+0.001) 
#            ut=fft(Signal)
#            utn=ut + 1*(np.random.uniform(0.1, 2, ut.shape)+np.sqrt(-1)*np.random.uniform(0.1, 2, ut.shape))
#            Signal=ifft(utn)
            Dataset_This_File = pd.DataFrame(columns=('Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum'),index = range(180))
            for segment_index in range(180):
               
               start_ind = segment_index*int(fs * Segment_length)
               
               stop_ind = (segment_index+1)*int(fs * Segment_length) 
               Segment = Signal[range(start_ind,stop_ind)]
               #Normal breathing                 
               if segment_index>2 and segment_index<60:                  
                  Segment = addsway(Segment, 0.5, 0.0125, len(Segment))
#               if segment_index>68 and segment_index<119:                  
#                  Segment = addsway(Segment, 0.5, 0.0125, len(Segment))
#               if segment_index>134 and segment_index<180:                  
#                  Segment = addsway(Segment, 0.5, 0.0125, len(Segment))
#               Stop breathing
#               if segment_index>118 and segment_index<132:
#                  Segment = addsway(Segment, 1, 0.0125, len(Segment))
#
#               if (segment_index>2 and segment_index<60) or (segment_index>68 and segment_index<119) or (segment_index>134 and segment_index<180):                  
#                  Segment = addsway(Segment, 0.5, 0.0125, len(Segment))

#
#               if segment_index>2 and segment_index<60:                  
#                  Segment = addsway(Segment, 1, 0.0125, len(Segment))
#                  
#               elif segment_index>68 and segment_index<119:
#                    Segment = addsway(Segment, 1, 0.0125, len(Segment))
#                    
#               elif segment_index>134 and segment_index<180:
#                    Segment = addsway(Segment, 1, 0.0125, len(Segment))
                    
                  
               Normalized_Segment = (Segment-np.mean(Segment))/np.std(Segment)
                               
#               yy,yy2=AMI(Segment)
               
               RPS = np.vstack([Normalized_Segment[range(int(RPS_Tau*fs),len(Segment))], Normalized_Segment[range(len(Segment)-int(RPS_Tau*fs))]])            
               RPS = RPS.T
               var = np.var(Segment)
               # Create linear regression object
               regr = linear_model.LinearRegression()
               X = RPS[:,0].reshape(len(RPS),1)
               Y = RPS[:,1].reshape(len(RPS),1)
               # Train the model using the training sets
               regr.fit(X,Y)
               Reg_Coeff = regr.coef_[0,0]
               Reg_Res_Sum = np.sum((regr.predict(X) - Y )**2)/var
               # Explained variance score: 1 is perfect prediction
               Reg_Var = regr.score(X, Y)
               
               Segment_mv = movingaverage(Normalized_Segment,10)
               f_welch, S_xx_welch = scipy.signal.welch(Segment_mv, fs=fs,nperseg=256)
#               plt.plot(f_welch, 10*np.log(S_xx_welch), '-s')
               
               df_welch = f_welch[1] - f_welch[0]
               PSD =  np.sum(S_xx_welch[range(4)])/np.sum(S_xx_welch[range(20)])
               PSD_Ratio = np.sum(S_xx_welch[range(6)])/np.sum(S_xx_welch[range(20)])
               
              
               Dataset_This_File.iloc[segment_index] = [Segment,var,PSD,PSD_Ratio, Reg_Coeff,Reg_Var, Reg_Res_Sum]
              
            Dataset = pd.concat([Dataset,Dataset_This_File],keys=['Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum'],ignore_index=True)


Dataset2 = pd.read_csv('RadarFeatures_label.csv', sep=',', na_filter=True)
Labels=Dataset2['label']#se = pd.Series(Lables)
Dataset['label'] = Labels.values
            

#Dataset.columns = ['Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum','label']
#Dataset = pd.concat([Dataset,tt],keys=['Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum',,'label'],ignore_index=True)
time_start = time.clock() 

indices_to_replace_2  = Dataset[((Dataset.label ==2))].index
Binary_target = [str(Dataset.label.iloc[i]) for i in range(len(Dataset))]

Features_names =['PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum']
Dataset[Features_names] = Dataset[Features_names].astype(float)
Features_values= Dataset[Features_names].values

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_predict
gnb = GaussianNB()


predicted = cross_val_predict(gnb, Features_values[:,[1, 3, 4]], Binary_target, cv=30)  #range(1,7)  only five gives a very good classification'''
str_pred = predicted  

time_elapsed = (time.clock() - time_start)

predicted = [int(predicted[i][0]) for i in range(len(predicted))]
Dataset['predicted']= pd.Series(predicted)
AA = 100*sum(Dataset.label==Dataset.predicted)/len(Dataset)     
print('Accuracy = %f' %AA)
AA = 100*sum(Dataset.label[Dataset.predicted==2]==2)/sum(Dataset.label==2)
print('Accuracy of Stop = %f' %AA)
AA = 100*sum(Dataset.label[Dataset.predicted==1]==1)/sum(Dataset.label==1)
print('Accuracy of Erratic = %f' %AA)
AA = 100*sum(Dataset.label[Dataset.predicted==0]==0)/sum(Dataset.label==0)
print('Accuracy of Normal = %f' %AA)

# Compute confusion matrix
cm = confusion_matrix(Binary_target, str_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

#plt.plot(Signal)
#plt.plot(Signal2)
#plt.plot(Signal2-Signal)
#ro=Dataset['PSD_Ratio']
#plt.plot(ro)   
#plt.xlabel('Segment_Index')
#plt.ylabel('PSD_Ratio')
#plt.savefig('psd.png', bbox_inches='tight')
#       
#arr = []
#arr2 = []
#for row in Dataset['Signal']:
#    val = (row-np.mean(row))/np.std(row)
#    yy1,yy2=AMI(val)
#    arr.append(yy1)
#    arr2.append(yy2)
#    t = np.vstack(arr)
#    t2 = np.vstack(arr2)
#    
#avg = t.mean(axis=0)
#avg2 = t2.mean(axis=0)
#
#f2 = plt.figure()
#f2, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=True)
#ax1.plot(range(len(avg)),avg)
#ax1.set_title('')
#ax2.plot(range(len(avg2)), avg2, color='g')
#f2.subplots_adjust(hspace=0)
#plt.setp([a.get_xticklabels() for a in f2.axes[:-1]], visible=False)