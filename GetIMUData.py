#!/usr/bin/env python
# coding: utf-8


import socket

import struct
import numpy as np
import time
import csv
import sys
import select
import time

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


DEBUG = False
num_of_windows = 100
num_freq_bins=10
num_time_bins=1
fs = 48000
TYPE = "BIN"
X = []
Y = []

warnings.simplefilter('ignore')


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



# Featurize each subject's gaze trajectory in the file
# This method is run once for each video file for each condition
def featurize_input(sound, window=False, window_num=1):
    try:
        temp = np.vstack(np.asarray(row) for row in sound[:-1])
        # print("shape"+str(temp.shape))
#         print(temp)
        sound = np.asfarray(temp[:-3], dtype='float')       
        # print(sound)
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
    except:
        # print("EXCEPTION")
        return np.array([])
















s = None

CIRCLE = 0
SQUARE = 1
LINE = 2
NOTHING = 3
import pickle

def get_action(input):
    if input is None or input.shape[0] == 0:
        return NOTHING
    with open("finalized_model.ml", 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model.predict(input)

readable = [] # list of readable sockets.  s is readable if a client is waiting.
content = bytearray(b"")


def start_imu():
    global s
    global readable
    s = socket.socket()         

    s.bind(('0.0.0.0', 8090 ))
    s.listen(0)         
    readable = [] # list of readable sockets.  s is readable if a client is waiting.


def get_imu_data():
    global content
    global readable
    global s



    content = bytearray(b"")
    socket_list = [sys.stdin, s]
    # Get the list sockets which are readable
    r,w,e = select.select(socket_list,[],[],0)
    for rs in r: # iterate through readable sockets
        if rs is s: # is it the server
            # client, addr = s.accept()
            # content = bytearray(b"")
            print('\rconnected')
            readable.append(s) # add the client
        
    for rs in readable:
        if rs is s:
            client, addr = s.accept()
            content = bytearray(b"")
            while True:
                rec = client.recv(32)
                content = content + rec
                if len(rec) ==0:
                    print('\rdisconnected')
                    readable.remove(rs)
                    client.close()
                    break
    content = str(content, "utf-8")
#     print(content)
    content = csv.reader(content.split('\n'))
#     print((list(content)))
    
    # print("Closing connection")
    
    # print("got data") 
    return(np.array(list(content)))
               
        




    # read_sockets, write_sockets, error_sockets = select.select(socket_list , [], [], 0)
    # print(len(read_sockets))
    # for sock in read_sockets:
    # #incoming message from remote server
    #     if sock == s:
    #         client, addr = s.accept()
    #         content = bytearray(b"")
    #         while True:
    #             rec = client.recv(32)
    #             content = content + rec
    #             if len(rec) ==0:
    #                 break
            
                         







if __name__ == "__main__":
    start_imu()
    while(1):
        # print("trying for data\n")
        data = get_imu_data()
        # print(data)
        print("ACTION"+str(get_action(featurize_input(data))))
        time.sleep(0.1)




