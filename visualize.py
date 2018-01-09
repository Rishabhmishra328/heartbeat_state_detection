import scipy.io.wavfile as wv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from progress.bar import Bar
import math
import csv

prediction_value = 100

def visualize(fpath):
    audio_file = wv.read(fpath, mmap = False)
    df_out = pd.read_csv('output.csv')
    df_out = df_out.loc[df_out['fpath'] == './data/set_a/__201101152256.wav']
    file_length = audio_file[1]
    prev_type = ''
    output = []
    for c, l, s, f in zip(df_out['confidence'],df_out['location'],df_out['sound'],df_out['fpath']):
        if not (prev_type == s):
            output.append([l,s])
        prev_type = s

    # print(output)

    with open("out.csv",'w') as resultFile:
            wr = csv.writer(resultFile)
            wr.writerow(['location','sound'])
            wr.writerows(output)       

    end_location = []
    for i in range(len(output)):
        end_location.append(output[i][0])
    end_location = end_location[1:]

    # print(end_location)
    prediction_curve = []
    for index in range(len(output[:-1])):
        # print(output[index][0])
        value = prediction_value if output[index][1] == 'S1' else -prediction_value
        start_loc = output[index][0]
        end_loc = end_location[index]
        prediction_curve[start_loc:end_loc] = [value] * (end_loc - start_loc)
    # prediction_curve[len(prediction_curve):file_length] = prediction_value if output[-1][1] == 'S1' else -prediction_value
    # print(prediction_curve)
    end = end_location[-1]
    plt.plot(audio_file[1][:end])
    plt.plot(prediction_curve)
    plt.show()

    # for index in range(file_length):
    #     prediction_curve[i] = 'S1' if 


visualize('./data/set_a/Aunlabelledtest__201101152256.wav')