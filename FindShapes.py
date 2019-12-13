#!/usr/bin/env python
# coding: utf-8

# # Using Sound to Detect Activities and Events

# ### Importing libraries

# In[5]:


#Feel free to add any libraries you need to here
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io.wavfile as wavfile
from scipy import signal
import cv2
import math
import copy
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loads the sound files
# 

# In[6]:


DEBUG = False
num_of_windows = 100
num_freq_bins=10
num_time_bins=1
fs = 48000
TYPE = "BIN"
X = []
Y = []
warnings.simplefilter('ignore')


# In[7]:


import glob

def load_actual_sound(directory, i):
    filename = i
    test_sound = np.load(filename)
    return test_sound

def load_sound_data(directory, window=False, num_of_windows=1, run_type="BIN"):
    if run_type == "BIN":
        return load_sound_data_bins(directory, window, num_of_windows)
    else:
        return load_sound_data_fv(directory, window, num_of_windows)

def load_sound_data_fv(directory, window=True, num_of_windows=1):
    fv = []
    print(directory + "... ", end=" ")
    for i in glob.glob('shapes/' + directory + '/' + '*.npy'):
        filename = i
        print(filename)
        temp = np.load(filename, allow_pickle=True)
        if (temp.shape[0]<10): continue
        temp = np.vstack(np.asarray(row) for row in temp[:-1])
        print("shape"+str(temp.shape))
#         print(temp)
        test_sound = np.asfarray(temp[:-3], dtype='float')        
        if window:
            windows = windowize(test_sound, num_of_windows)
            if DEBUG: print(windows.shape)
            temp = []
            for sound in windows:
                fv_new = featurize_input(sound)
                temp.append(copy.deepcopy(np.array(fv_new[0,:])))
            if DEBUG: print(len(temp))
            
            fv.append(copy.deepcopy(temp))
        else:
            fv_new = featurize_input(test_sound)
            fv.append(fv_new[0,:])
    return np.array(fv)

def load_sound_data_bins(directory, window=False, num_of_windows=1):
    fv = []
    print(directory + "... ", end=" ")
    for i in glob.glob('shapes/' + directory + '/' + '*.npy'):
        filename = i
        print(filename)
        temp = np.load(filename, allow_pickle=True)
        if (temp.shape[0]<10): continue
        temp = np.vstack(np.asarray(row) for row in temp[:-1])
        print("shape"+str(temp.shape))
#         print(temp)
        test_sound = np.asfarray(temp[:-3], dtype='float')
        if window:
            windows = windowize(test_sound, num_of_windows)
            if DEBUG: print(windows.shape)
            temp = []
            for sound in windows:
                fv_new = binify(spectrogram(sound))
                temp.append(copy.deepcopy(np.array(fv_new[0,:])))
            if DEBUG: print(len(temp))
            
            fv.append(copy.deepcopy(temp))
        else:
            fv_new = binify(spectrogram(test_sound))
            fv.append(fv_new[0,:])
    print(np.array(fv).shape)
    return np.array(fv)


# ## Extract Features
# 

# In[8]:


def freq_to_note(freq):
    if (freq == 0):
        return 0
    return 69+12*math.log(freq/440,2)

# Calculate the difference between a current point and the short term average
# of previous points for each point and square it
# Returns the average of those (differences^2)
f,t = 0,0

def spectrogram(sound, fmin = 1, fmax = 10000):
    f,t,ppx = signal.spectrogram(sound, nperseg=len(sound)-1, fs=fs, noverlap=0)
    #https://stackoverflow.com/questions/48097164/limiting-scipy-signal-spectrogram-to-calculate-only-specific-frequencies
    freq_range = np.where((f>=fmin) & (f<=fmax))
    f = f[freq_range]
    ppx = ppx[freq_range,:][0]
    return ppx

def binify(ppx, window=False):
    return np.transpose(cv2.resize(ppx[:,:],(num_time_bins, num_freq_bins)))

def first_freq(ppx):
    return np.argmax(ppx)

def second_freq(ppx, width=100):
    ff = first_freq(ppx)
    if (ff-width>0):
        lower = ppx[:ff-width]
        if (ff+width>=len(ppx)):
            return np.argmax(lower)
        else:
            upper = ppx[ff+width:]
            i_l = np.argmax(lower)
            i_u = np.argmax(upper)
            if (lower[i_l]>upper[i_u]):
                return i_l
            return i_u
    else:
        upper = ppx[ff+width:]
        return np.argmax(upper)
    

def windowize(sound, window_num=100):
    window_size = int(len(sound)//window_num)
    beginning_index = 0
    new_sound = []
#     new_sound = sound[int(beginning_index):int(beginning_index+window_size)]
#     beginning_index = int(window_size//2)
    while((beginning_index + window_size) <= len(sound)):
        new_sound.append(sound[int(beginning_index):int(beginning_index+window_size)])
        beginning_index = beginning_index + (window_size//2)
    new_sound = np.array(new_sound)
    if DEBUG: print("windowed:" + str(new_sound.shape))
    return new_sound
    
# Calculate the difference between a current point and the short term average
# of previous points for each point
# Returns the average of those differences
def average_amplitude(sound):
    return np.average(sound)


# In[9]:


# Featurize each subject's gaze trajectory in the file
# This method is run once for each video file for each condition
def featurize_input(sound, window=False, window_num=1):
    out = []
    # Where i is each subject in the file
    fv = []
    ppx = spectrogram(sound)
    #added a feature that calculates a difference between a point and the short-term mean
    #add your features to fv (each feature here should be a single number)
#     fv.append(freq_to_note(first_freq(ppx)))
#     fv.append(freq_to_note(second_freq(ppx)))
    fv.append(average_amplitude(sound))
    fv.append(np.max(sound))
    out.append(fv)
    out = np.array(out)
#     print(out.shape)
    return out


# In[10]:


def generate_data(window=False, run_type="BIN"):
    global X
    global Y
    X_Blender = load_sound_data("circles", window, num_of_windows, run_type)
    X_Microwave = load_sound_data("squares", window, num_of_windows, run_type)
    X_Music = load_sound_data("lines", window, num_of_windows, run_type)


    if DEBUG: print (X_Blender.shape,X_Microwave.shape,X_Music.shape,X_Siren.shape,X_Vac.shape)

    #Assigning groundtruth conditions to each participant. 
    if window:
        Y_Blender = [0.0] * len(X_Blender[0]) 
        Y_Microwave = [1.0] * len(X_Microwave[0]) 
        Y_Music = [2.0] * len(X_Music[0]) 
        ###
        Y_Blender = [Y_Blender] * len(X_Blender) 
        Y_Microwave = [Y_Microwave] * len(X_Microwave) 
        Y_Music = [Y_Music] * len(X_Music) 

    else:
        Y_Blender = [0.0] * len(X_Blender) 
        Y_Microwave = [1.0] * len(X_Microwave) 
        Y_Music = [2.0] * len(X_Music) 




    X = np.concatenate((X_Blender,X_Microwave,X_Music)) 
    Y = np.concatenate((Y_Blender,Y_Microwave,Y_Music))
    print("x,y shapes")
    print(X.shape)
    print(Y.shape)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# # Train and Score

# In[18]:


import pickle
def train_model(window=False):
#     global X
#     global Y
    clf = RandomForestClassifier(n_estimators=10, max_depth=4,random_state=0)
    print("\nRandom Forest accuracy: ", end=" ")
    scores_training = []
    cv_training = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv_training.split(X):
#         print("Train Index: ", train_index)
#         print("Test Index: ", test_index)
        if window:
            X_train, X_test, y_train, y_test = [],[],[],[]
            for i in train_index:
                X_train.extend(X[i])
                y_train.extend(Y[i])
            for i in test_index:
                X_test.extend(X[i])
                y_test.extend(Y[i])            
        else:
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        clf.fit(X_train, y_train);
        scores_training.append(clf.score(X_test, y_test))
    print(str(np.mean(scores_training)))
    filename = 'finalized_model.ml'
    pickle.dump(clf, open(filename, 'wb'))
    return np.mean(scores_training)
        


# In[19]:



print("\nFeatured and not windowed:")
generate_data(window=False, run_type="FEAT")
train_model()
# print("\nFeatured and windowed:")
# generate_data(window=True, run_type="FEAT")
# train_model(window=True)
# print("\nBinned and not windowed:")
# generate_data(window=False, run_type="BIN")
# train_model()
# print("\nBinned and windowed:")
# generate_data(window=True, run_type="BIN")
# train_model(window=True)


# # Generate Frequency Plots

# In[4]:




cmap=plt.cm.bone
cmap.set_under(color='k', alpha=None)

plt.title("Blender")
f,t,pxx = signal.spectrogram(load_actual_sound("circles",1), nperseg=1024, fs=fs, noverlap=1024/2)
plt.pcolormesh(np.log10(pxx[0:100,:]),cmap=cmap)
plt.show()

plt.title("Microwave")
f,t,pxx = signal.spectrogram(load_actual_sound("squares",1), nperseg=1024, fs=fs, noverlap=1024/2)
plt.pcolormesh(np.log10(pxx[0:100,:]),cmap=cmap)
plt.show()

plt.title("Music")
f,t,pxx = signal.spectrogram(load_actual_sound("lines",1), nperseg=1024, fs=fs, noverlap=1024/2)
plt.pcolormesh(np.log10(pxx[0:100,:]),cmap=cmap)
plt.show()


# In[ ]:




