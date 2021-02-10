import numpy as np
import tensorflow as tf

import pandas as pd
import math
from scipy import special
import scipy
import random as rd
import function as fc
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from function_class import func
from DQN_NET import DeepQNetwork

np.random.seed(2)

trainM = 160000000
testM=10000000
f = func()
f.Bandwidth = 720e3
f.intial()
s = f.getstate()
RL = DeepQNetwork()
ca = f.ca
back = f.back
step = 0
reward_sample = 0
reward_slot = []
reward_random = []
reward_policy = []
transmit_in0 = np.zeros((125, 68))
transmit_action0 = np.zeros((125, 8))
# 数据统计 125（5种图片大小，5种置信度，5中等待时间） 68action维度
transmit_in1 = np.zeros((125, 68))
transmit_action1 = np.zeros((125, 8))
randomno = 0
table=sio.loadmat('store_CRF0604.mat')['store_CRF']
picindex=np.zeros(3)
# 数据统计 125（5种图片大小，5种置信度，5中等待时间） 68action维度

def countnum(transmit_in0, transmit_in1, s, a, In, action, transmit_action0, transmit_action1):
    if In == 0:
        for i in range(0, 3):
            size = s[i * 4]
            yita = s[i * 4 + 1]
            T_wait = s[i * 4 + 2]
            if size != 0 or yita != 0:
                S, Yita, Tw = classi(size, yita, T_wait)
                index = Tw * 25 + Yita * 5 + S
                transmit_in0[index, int(a)] += 1
                transmit_action0[index, int(action[i] + 1)] += 1
                if action[i] == 1:
                    transmit_action0[index, int(action[f.n_ca] + 3)] += 1
    else:
        for i in range(0, 3):
            size = s[i * 4]
            yita = s[i * 4 + 1]
            T_wait = s[i * 4 + 2]
            if size != 0 or yita != 0:
                S, Yita, Tw = classi(size, yita, T_wait)
                index = Tw * 25 + Yita * 5 + S
                transmit_in1[index, int(a)] += 1
                transmit_action1[index, int(action[i] + 1)] += 1
                if action[i] == 1:
                    transmit_action1[index, int(action[f.n_ca] + 3)] += 1
    return transmit_in0, transmit_in1, transmit_action0, transmit_action1


def classi(size,yita,T_wait):
    if size<=0.1782:
        S=0
    elif size<=0.3666:
        S=1
    elif size <= 0.5669:
        S=2
    elif size <= 0.7725:
        S=3
    else:
        S=4

    if yita<=0.1117:
        Yita=0
    elif yita<=0.1319:
        Yita=1
    elif yita <= 0.2177:
        Yita=2
    elif yita <= 0.3141:
        Yita=3
    else:
        Yita=4

    if T_wait<=0.1:
        Tw = 0
    elif T_wait<=0.2:
        Tw=1
    elif T_wait <=0.3:
        Tw=2
    elif T_wait <=0.4:
        Tw=3
    else:
        Tw = 4
    return S,Yita,Tw


for t in range(0, trainM):
    for pi in range(0,3):
        picindex[pi]=f.X[pi*2,9]
    num_index = sum(f.x_num)
    if num_index != 0:
        print('time:', t)
        a = RL.choose_action(s)
        In = s[len(s) - 1]
        print(In)
        print(f.I_pointer)
        action=f.atoaction(In,a,s,table,picindex)
        print(action)

    else:
        action = [-1, -1, -1, 0]

    r,r_test=f.multiple_rechan(action)
    reward_sample+=r_test
    f.twa()
    f.intial()
    snext = f.getstate()
    if num_index != 0:
        if f.count == 1:
            transmit_in0, transmit_in1, transmit_action0, transmit_action1 = countnum(transmit_in0, transmit_in1, s, a,
                                                                                      In, action, transmit_action0,
                                                                                      transmit_action1)
        RL.store_transition(s, a, r, snext)
        if (step > 200) and (step % 200 == 5):
            RL.learn()
        step += 1
    s = snext
    if f.epi == 1:
        reward_slot.append(reward_sample)
        if randomno == 0:
            reward_random.append(reward_sample)
        if f.count == 1:
            reward_policy.append(reward_sample)
        reward_sample = 0
        f.epi = 0

    if t == 100000000:
        randomno = 1
        RL.lr = 1e-5
        RL.greedy = 0.95

    if t == 150000000:
        f.count = 1
        RL.greedy = 1

acc = f.acc
pic = f.pic
trans = f.trans
transnot = f.transnot
accback = f.accback
picback = f.picback
acc_all = acc / pic
delay = trans / pic
delay_ca = transnot / picback
delay_back = (trans - transnot) / (pic - picback)
acc_ca = accback / picback
acc_back = (acc - accback) / (pic - picback)
reward_random_ave = np.mean(reward_random)
reward_policy_ave = np.mean(reward_policy)

print('acc:', acc)
print('pic:', pic)
print('trans:', trans)
print('accback:', accback)
print('picback:', picback)
print('transnot:', transnot)
print('accall:', acc_all)
print('delay:', delay)
print('delay_ca:', delay_ca)
print('delay_back:', delay_back)
print('acc_ca:', acc_ca)
print('acc_back:', acc_back)
print('reward_random_ave:', reward_random_ave)
print('reward_policy_ave:', reward_policy_ave)
sio.savemat('transmit_in0.mat', {'transmit_in0': transmit_in0})
sio.savemat('transmit_in1.mat', {'transmit_in1': transmit_in1})
sio.savemat('transmit_action0.mat', {'transmit_action0': transmit_action0})
sio.savemat('transmit_action1.mat', {'transmit_action1': transmit_action1})
sio.savemat('reward.mat', {'reward': reward_slot})


#test 过程
reward_test = []
f2 = func()
f2.Bandwidth = 700e3
f2.intial()
f2.testindex=1
reward_sample = 0
transmit_in0 = np.zeros((125, 68))
transmit_action0 = np.zeros((125, 8))
# 数据统计 125（5种图片大小，5种置信度，5中等待时间） 68action维度
transmit_in1 = np.zeros((125, 68))
transmit_action1 = np.zeros((125, 8))
s = f2.getstate()
for t in range(0, testM):

    num_index = sum(f2.x_num)
    for pi in range(0,3):
        picindex[pi]=f2.X[pi*2,9]
    if num_index != 0:
        print('time:', t)
        a = RL.choose_action(s)
        In = s[len(s) - 1]
        print(In)
        print(f2.I_pointer)
        action=f2.atoaction(In,a,s,table,picindex)
        print(action)
    else:
        action = [-1, -1, -1, 0]

    r,r_test=f2.multiple_rechan(action)
    reward_sample+=r_test
    f2.twa()
    f2.intial()
    snext = f2.getstate()
    if num_index != 0:
        if f2.count == 1:
            transmit_in0, transmit_in1, transmit_action0, transmit_action1 = countnum(transmit_in0, transmit_in1, s, a,
                                                                                      In, action, transmit_action0,
                                                                                      transmit_action1)
    s = snext
    if f2.epi == 1:
        reward_test.append(reward_sample)
        reward_sample = 0
        f2.epi = 0

    if t == 1:
        f2.count = 1


acc = f2.acc
pic = f2.pic
trans = f2.trans
transnot = f2.transnot
accback = f2.accback
picback = f2.picback
acc_all = acc / pic
delay = trans / pic
delay_ca = transnot / picback
delay_back = (trans - transnot) / (pic - picback)
acc_ca = accback / picback
acc_back = (acc - accback) / (pic - picback)
reward_ave = np.mean(reward_test)


print('acc:', acc)
print('pic:', pic)
print('trans:', trans)
print('accback:', accback)
print('picback:', picback)
print('transnot:', transnot)
print('accall:', acc_all)
print('delay:', delay)
print('delay_ca:', delay_ca)
print('delay_back:', delay_back)
print('acc_ca:', acc_ca)
print('acc_back:', acc_back)
print('reward_ave:', reward_ave)
sio.savemat('transmit_test_in0.mat', {'transmit_in0': transmit_in0})
sio.savemat('transmit_test_in1.mat', {'transmit_in1': transmit_in1})
sio.savemat('transmit_test_action0.mat', {'transmit_action0': transmit_action0})
sio.savemat('transmit_test_action1.mat', {'transmit_action1': transmit_action1})
sio.savemat('reward_test.mat', {'reward': reward_test})
