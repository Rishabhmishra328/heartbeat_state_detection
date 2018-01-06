import scipy.io.wavfile as wv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os

df_data = pd.read_csv('./data/set_a_timing.csv')
df_data[['fname']] = './data/' + df_data[['fname']]

fname = df_data['fname']
sound = df_data['sound']
location = df_data['location']

data_list = []

if not os.path.exists('./data/S1/'):
    os.makedirs('./data/S1/')
if not os.path.exists('./data/S2/'):
    os.makedirs('./data/S2/')


for i in range(len(fname)):
    temp = {}
    temp['fname'] = fname[i]
    temp['sound'] = sound[i]
    temp['location'] = location[i]
    data_list.append(temp)

file_counter = 0
for data in data_list:
    audio_file = wv.read(data['fname'], mmap = False)
    output_file = './data/' + data['sound'] + '/' + str(data['fname'][13:]) + '_' + str(file_counter) + '.wav'
    signal_index = data['location']
    sample_audio_file = audio_file[1][signal_index-500 : signal_index+2500]
    wv.write(output_file, audio_file[0], sample_audio_file)
    file_counter += 1
