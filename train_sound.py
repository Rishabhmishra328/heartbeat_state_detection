import scipy.io.wavfile as wv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from progress.bar import Bar
import math

def sigmoid(x):
    max_val = 0
    for val in x:
        if math.fabs(val)>max_val:
            max_val = math.fabs(val)

    res = []
    for val in x:
        # if val <-2:
        #     val = -2
        # new_val = ((val/max_val)+2)/3
        res.append(val/max_val)

    res = np.asarray(res, dtype=np.float32)
    return res

#Data preperation
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
data_pbar = Bar('Audio preperation progress', max = len(data_list))

input_data = []
output_data = []

for data in data_list:
    audio_file = wv.read(data['fname'], mmap = False)
    output_file = './data/' + data['sound'] + '/' + str(data['fname'][13:-5]) + '_' + str(file_counter) + '.wav'
    signal_index = data['location']
    sample_audio_file = np.asarray(audio_file[1][signal_index-500 : signal_index+2500], dtype= np.float32)
    sample_audio_file = sigmoid(sample_audio_file)
    input_data.append(sample_audio_file)
    output_data.append(np.asarray([1.0,0.0], dtype=np.float32) if data['sound'] == 'S1' else np.asarray([0.0,1.0], dtype= np.float32))
    wv.write(output_file, audio_file[0], sample_audio_file)
    file_counter += 1
    data_pbar.next()

# print(input_data, output_data)
#Training
# params
learning_rate = 0.001
epochs = 100
display_step = 5

input_data = np.array(input_data, dtype=np.float32)
output_data = np.array(output_data, dtype=np.float32)

# test = []
# a_pbar = Bar('Preparing graph', max = len(input_data)*3000)
# for g in input_data:
#     for m in g:
#         test.append(m)
#         a_pbar.next()

# test = np.asarray(test, dtype=np.float32)
# print(test)

# print(input_data.shape)

def train():
    x = tf.placeholder(tf.float32, [None, input_data.shape[1]])
    y = tf.placeholder(tf.float32, [None, output_data.shape[1]])

    #model
    W = tf.Variable(tf.zeros([input_data.shape[1],output_data.shape[1]]))
    b = tf.Variable(tf.zeros([output_data.shape[1]]))
    #linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)
    #cross entropy
    cost_function = -tf.reduce_sum(y * tf.log(model))
    #gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    #initializing
    init = tf.global_variables_initializer()

    #firing up session
    with tf.Session() as sess:
        sess.run(init)
        #training
        avg_cost = 0.
        for iteration in range (epochs):
            #fitting data
            sess.run(optimizer, feed_dict={x: input_data, y: output_data})
            #calculating total loss
            avg_cost += sess.run(cost_function, feed_dict={x: input_data, y: output_data})/epochs
            #display logs each iteration
            if iteration % display_step == 0 :
                print('Iteration : ', '%04d' % (iteration + 1), 'cost = ', avg_cost)

        print('Training complete')

        # #testing
        # predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) 
        # #accuracy
        # accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        # print('Accuracy : ', accuracy.eval({x: test_input_data, y: test_output_data}), 'Predictions : ', predictions.eval({x: test_input_data, y: test_output_data}))

train()

