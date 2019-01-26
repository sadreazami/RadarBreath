
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
from scipy.stats.stats import pearsonr 
import seaborn as sns
from scipy import stats
from sklearn import tree
from sklearn.cross_validation import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
plt.close('all')

def ReadData(filename):
    df = pd.read_csv(filename, sep=',', na_filter=True)
    data = df.values
    return data
    
def FindZone(x):
    ZoneSum = x.sum(axis=0)
    ZoneNum = zip(*ZoneSum.nonzero())
    return ZoneNum[0][0]
    
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma  
    
dic = {'SUB': ['S02_','S03_','S04_'], 'POS':['SIT_'],'Rep':['R1_','R2_','R3_']}
  #',,'S04_'

DownSampleRate = 10
fs = 1480/DownSampleRate
RPS_Tau = 0.05 #sec
RPS_Dim = 2
Segment_length = 5 # 10 seconds of signal
numberOfRows = 180
# create dataframe
Dataset = pd.DataFrame(columns=('Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum','label'))

for subject in dic['SUB']:
    for posture in dic['POS']:
        for repitition in dic['Rep']:
            filename = 'CSVFiles/'+subject+posture+repitition+'RADAR.csv'
            print(filename)
            x = ReadData(filename)
            ZoneNum = FindZone(x)
            Signal = x[:,ZoneNum]
            Signal = scipy.signal.decimate(Signal,DownSampleRate )
            if len(Signal)>=fs*numberOfRows*Segment_length:
               Signal = np.array(Signal[range(fs*numberOfRows*Segment_length)])
            else:
               Signal = np.concatenate((Signal,np.zeros(fs*numberOfRows*Segment_length-len(Signal))),axis = 0)
            #print(len(Signal)/fs)
            LabelFilename = 'CorrectedLabels/'+subject+posture+repitition+'label.csv'
            Labels = pd.read_csv(LabelFilename, sep=',', na_filter=True,header=None)
            Labels = Labels[0].tolist()
            Dataset_This_File = pd.DataFrame(columns=('Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum','label'),index = range(180))
            for segment_index in range( 180):
               
               start_ind = segment_index*int(fs * Segment_length)
               '''if (segment_index==179) and (len(Signal<fs * Segment_length*180)):
                  stop_ind = len(Signal) 
               else:'''
               stop_ind = (segment_index+1)*int(fs * Segment_length) 
               Segment = Signal[range(start_ind,stop_ind)]
               Normalized_Segment = (Segment-np.mean(Segment))/np.std(Segment) 
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
               #S_xx_welch = (S_xx_welch)/np.max(S_xx_welch)
                # Integrate PSD over spectral bandwidth
                # to obtain signal power `P_welch`
               df_welch = f_welch[1] - f_welch[0]
               PSD =  np.sum(S_xx_welch[range(4)])/np.sum(S_xx_welch[range(20)])
               PSD_Ratio = np.sum(S_xx_welch[range(6)])/np.sum(S_xx_welch[range(20)])
               
               '''d = {'Signal': Segment,
                    'var':var,
                    'PSD':PSD,
                    'PSD_Ratio':PSD_Ratio,
                    'RegCoeff':Reg_Coeff,
                    'RegVar':Reg_Var,
                    'RegResSum':Reg_Res_Sum,
                    'label':Labels[segment_index]}
               df = pd.DataFrame(data = d)'''
                                    
               Dataset_This_File.iloc[segment_index] = [Segment,var,PSD,PSD_Ratio, Reg_Coeff,Reg_Var, Reg_Res_Sum,Labels[segment_index]]
               #Dataset_This_File= pd.concat([Dataset_This_File,df],keys=['Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum','label'],ignore_index=True)                   
               #Dataset_This_File.Signal.iloc[segment_index]=Segment
            Dataset = pd.concat([Dataset,Dataset_This_File],keys=['Signal','var','PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum','label'],ignore_index=True)

indices_to_replace_2  = Dataset[((Dataset.label ==2))].index
Binary_target = [str(Dataset.label.iloc[i]) for i in range(len(Dataset))]
'''for i in indices_to_replace_2:
   Binary_target[i] ='1'''
   
sns.set(font_scale=3)
ax = plt.figure()
#ax = sns.distplot(Dataset['PSD'][Dataset['label']==2],hist = False, color="r",label ='Stop') 
ax = sns.distplot(Dataset['PSD'][Dataset['label']!=0],hist = False, color="b",label ='NotNormal') 
ax = sns.distplot(Dataset['PSD'][Dataset['label']==0],hist = False, color="g",label ='Normal') 
handles, labels = ax.get_legend_handles_labels() 
ax.legend(handles, labels,fontsize = 20)
plt.show()   



'''ax = plt.figure()
ax = sns.distplot(Dataset['PSD_Ratio'][Dataset['label']==2],hist = False, color="r",label ='Stop') 
ax = sns.distplot(Dataset['PSD_Ratio'][Dataset['label']==1],hist = False, color="b",label ='Erratic') 
ax = sns.distplot(Dataset['PSD_Ratio'][Dataset['label']==0],hist = False, color="g",label ='Normal') 
handles, labels = ax.get_legend_handles_labels() 
ax.legend(handles, labels,fontsize = 20)
plt.show()   

ax = plt.figure()
ax = sns.distplot(Dataset['RegCoeff'][Dataset['label']==2],hist = False, color="r",label ='Stop') 
ax = sns.distplot(Dataset['RegCoeff'][Dataset['label']==1],hist = False, color="b",label ='Erratic') 
ax = sns.distplot(Dataset['RegCoeff'][Dataset['label']==0],hist = False, color="g",label ='Normal') 
handles, labels = ax.get_legend_handles_labels() 
ax.legend(handles, labels,fontsize = 20)
plt.show()   

ax = plt.figure()
ax = sns.distplot(Dataset['RegVar'][Dataset['label']==2],hist = False, color="r",label ='Stop')
ax = sns.distplot(Dataset['RegVar'][Dataset['label']==1],hist = False, color="b",label ='Erratic') 
ax = sns.distplot(Dataset['RegVar'][Dataset['label']==0],hist = False, color="g",label ='Normal') 
handles, labels = ax.get_legend_handles_labels() 
ax.legend(handles, labels,fontsize = 20)
plt.show() '''


ax = plt.figure()
ax = sns.distplot(Dataset['RegResSum'][Dataset['label']==2],hist = False, color="r",label ='Stop') 
ax = sns.distplot(Dataset['RegResSum'][Dataset['label']!=2],hist = False, color="b",label ='NotStop') 
#ax = sns.distplot(Dataset['RegResSum'][Dataset['label']==0],hist = False, color="g",label ='Normal') 
handles, labels = ax.get_legend_handles_labels() 
ax.legend(handles, labels,fontsize = 20)
plt.show()


#Dataset = Dataset[Dataset.label!=3]
#Dataset.set_index(range(len(Dataset)), inplace = True)
Features_names =['PSD','PSD_Ratio','RegCoeff','RegVar','RegResSum']
Dataset[Features_names] = Dataset[Features_names].astype(float)
Features_values= Dataset[Features_names].values

grouped = Dataset[Features_names].groupby(Dataset.label) 


'''from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
#predicted = clf.fit(Exclude_start[:,range(5)], Exclude_start[:,5]).predict(Exclude_start[:,range(5)])
predicted = cross_val_predict(clf, Features_values, Binary_target, cv=30)'''

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
predicted = cross_val_predict(gnb, Features_values[:,[0,3,4]], Binary_target, cv=30)  #range(1,7)  only five gives a very good classification'''


'''for i in range(1,len(predicted)):
   if (predicted[i] == '0') and ((Features_values[i][1]<0.9)): #and (Features_values[3][0]<0.1):
      predicted[i] = '1'''
  
   #if (predicted[i] == '1') and (Features_values[i][0]<0.7)
      
'''for i in range(1,len(predicted)-2):
   if (predicted[i] == '1') and (Features_values[i][4]>0.2):
      predicted[i] = '2'''
str_pred = predicted  

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


plt.figure()
color ='bkr'
t = 0               
for i in range(len(Dataset)):
    time = np.float64(range(t, t+len(Dataset.Signal.iloc[i])))/(fs)
    plt.plot(time,Dataset.Signal.iloc[i],color=color[Dataset.label.iloc[i]])
    t = t+len(Dataset.Signal.iloc[i])
plt.show()
color ='bkr'
plt.figure()        
t = 0               
for i in range (900,1080):#range(len(Dataset)):
    time = np.float64(range(t, t+len(Dataset.Signal.iloc[i])))/(fs*60)
    plt.plot(time,Dataset.Signal.iloc[i],color=color[predicted[i]])
    t = t+len(Dataset.Signal.iloc[i])
plt.xlabel('time (min)',fontsize = 40)    
plt.ylabel('Radar Signal',fontsize = 40)
plt.text(2, 11100, 'Normal', style='italic',color='white',
        bbox={'facecolor':'blue', 'alpha':1, 'pad':5})
plt.text(2, 10800, 'Erratic ', style='italic',color='white',
        bbox={'facecolor':'black', 'alpha':1, 'pad':5})
plt.text(2, 10500, 'Stop   ', style='italic',color='white',
        bbox={'facecolor':'red', 'alpha':1, 'pad':5})
plt.ylim([8000 ,12000])
plt.show()

plt.figure()
plt.scatter(Dataset['PSD_Ratio'], Dataset['RegCoeff'], label= Dataset['label'], c = color)


plt.figure()
plt.plot(Features_values[:,3])


DD1 = Dataset[(Dataset.label ==1)][(Dataset.predicted ==0) ]
DD2 = Dataset[(Dataset.label ==1)][(Dataset.predicted ==2) ]
Dataset.mean()


grouped = Dataset[Features_names].groupby(Dataset.label) 

#def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(Binary_targets))
#    plt.xticks(tick_marks, iris.target_names, rotation=45)
#    plt.yticks(tick_marks, iris.target_names)
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(Binary_target, str_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Normal','Er','St'], rotation=45)
plt.yticks(tick_marks, ['Normal','Er','St'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

'''# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()

dic = {'SUB': ['S04_'], 'POS':['SIT_'],'Rep':['R3_']}
Test_Sig= np.zeros([180,3])
Signal_test = np.zeros([180,5*fs])
count = -1
for subject in dic['SUB']:
    for posture in dic['POS']:
        for repitition in dic['Rep']:
            
            filename = 'CSVFiles/'+subject+posture+repitition+'RADAR.csv'
            print(filename)
            x = ReadData(filename)
            ZoneNum = FindZone(x)
            Signal = x[:,ZoneNum]
            Signal = scipy.signal.decimate(Signal,DownSampleRate )
            if len(Signal)>=fs*numberOfRows*Segment_length:
               Signal = np.array(Signal[range(fs*numberOfRows*Segment_length)])
            else:
               Signal = np.concatenate((Signal,np.zeros(fs*numberOfRows*Segment_length-len(Signal))),axis = 0)
            #print(len(Signal)/fs)
           
            
            for segment_index in range( 180):
               count = count +1
               start_ind = segment_index*int(fs * Segment_length)
               stop_ind = (segment_index+1)*int(fs * Segment_length) 
               Segment = Signal[range(start_ind,stop_ind)]
               Normalized_Segment = (Segment-np.mean(Segment))/np.std(Segment) 
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
               
               Segment_mv = movingaverage(Segment,10)
               f_welch, S_xx_welch = scipy.signal.welch(Segment_mv, fs=fs,nperseg=256)
               S_xx_welch = (S_xx_welch)/np.max(S_xx_welch)
                # Integrate PSD over spectral bandwidth
                # to obtain signal power `P_welch`
               df_welch = f_welch[1] - f_welch[0]
               PSD =  np.sum(S_xx_welch[range(4)])/np.sum(S_xx_welch[range(20)])
               PSD_Ratio = np.sum(S_xx_welch[range(6)])/np.sum(S_xx_welch[range(20)])
               
               Test_Sig[count,:] = [PSD,Reg_Var,Reg_Res_Sum]
               Signal_test[count,:]=Segment
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
predicted = gnb.fit(Features_values[:,[0,3,4]], Binary_target).predict(Test_Sig)  #range(1,7)  only five gives a very good classification

with open('CorrectedLabels/'+subject+posture+repitition+'label.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in predicted:
        writer.writerow([int(val)])   
        
plt.figure()
color ='grkb'
t = 0               
for i in range(len(Signal_test)-1):
    time = np.float64(range(t, t+5*fs))/fs
    plt.plot(time,Signal_test[i,:],color=color[int(predicted[i])])
    t = t+5*fs
plt.show()'''