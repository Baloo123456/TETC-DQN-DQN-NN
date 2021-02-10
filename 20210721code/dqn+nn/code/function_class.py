import numpy as np
import math
import scipy.io as sio
from scipy import special
from scipy import optimize
import scipy
import random as rd
import time
from scipy.integrate import tplquad,dblquad,quad

class func:
    def __init__(
            self,
            n_ca=3,
            n_CRF=5,
            gain=1,
            n_actions=20,
            n_features=13,     #置信度+等待时间+信道状况
            X = np.zeros((30,12)),
            I=np.zeros(6),
            pic_local=0,
            ca_p=np.load("ca_2000_c2.npy"),
            back_p=np.load("back0406.npy"),
            R=0,
            x_number=np.zeros(3),
            T_max=2,
            num_loss=0,
            epi=0,
            R_not=0,
            R_wait=0,
            input_the=1,
            T_tr=0,
            I_pointer=0,
            pic=0,
            acc=0,
            trans=0,
            count=0,
            transnot=0,
            accback=0,
            picback=0,
            channelgain=1,
            mean=1,
            tau=10e-3,
            bata=0.0024
    ):
        self.bata=bata
        self.n_ca=n_ca
        self.n_CRF=n_CRF
        self.gain=channelgain
        self.n_actions = n_actions
        self.n_features = n_features
        self.X=X
        self.I=I
        self.pic_local=pic_local
        self.ca=ca_p
        self.back=back_p
        self.R_tr=R
        self.R_wait=R_wait
        self.x_num=x_number
        self.T_max=T_max         #不回传图片收益算在第一次决策里
        self.loss=num_loss
        self.epi=epi
        self.R_not=R_not
        self.input_pro=input_the
        self.T_tr=T_tr
        self.I_pointer=I_pointer
        self.pic=pic
        self.acc=acc
        self.trans=trans
        self.count=count
        self.transnot=transnot
        self.accback=accback
        self.picback=picback
        self.gain=gain
        self.mean=mean
        self.tau=tau
        self.Bandwidth=0


    def twa(self):    #action是一个两个值的向量 （0,1,2,3,4）,（0,1,2,3,4,5）
        for i in range(0,self.n_ca):
            if self.x_num[i]>0:
                for j in range(0,int(self.x_num[i])):
                    index=i*self.T_max+j
                    self.X[index,3]+=1

        if self.I_pointer-1 == 0:
            self.I=np.zeros(6)
            self.I_pointer=0
        elif self.I_pointer!=0:
            self.I_pointer -= 1

    def intial(self):
        input_pro=np.random.rand(self.n_ca)
        for i in range(0,self.n_ca):
            if input_pro[i]<0.1:
                if self.x_num[i]<self.T_max:
                    index=i*self.T_max+int(self.x_num[i])
                    self.X[index,0]=self.ca[self.pic_local,2]
                    self.X[index, 1] = self.ca[self.pic_local, 0]
                    self.X[index, 5] = self.ca[self.pic_local, 1]
                    self.X[index, 2] = 0
                    self.X[index,3]=1
                    self.X[index, 4] = 0
                    self.X[index, 6] = 1
                    self.X[index, 7] = i
                    self.X[index, 8] = 0
                    self.X[index, 9] = self.pic_local
                    self.X[index, 10] = 0
                    self.X[index, 11] = 0
                    self.x_num[i] += 1
                # else:
                #     for z in range(0, self.T_max - 1):
                #         self.X[z, :] = self.X[z + 1, :]
                #     self.X[self.T_max - 1, :] = np.zeros(12)
                self.pic_local += 1
                if self.pic_local == 5999:
                    self.pic_local = 0
                    self.epi=1

    def getstate(self):
        s=np.zeros(self.n_features)
        for i in range(0,self.n_ca):
            if self.x_num[i]>0:
                index=i*self.T_max
                s_size,s_yita,s_T_wait=func.classfi(self,self.X[index,1], self.X[index,0],self.X[index,3])
                s[i*4]=s_size
                s[i*4+1]=s_yita
                s[i*4+2]=s_T_wait
                s[i*4+3]=self.x_num[i]
        s[self.n_features - 1] = self.I_pointer/10

        return s


    def classfi(self,size,yita,T_wait):

        s_size=size
        s_yita=yita

        if T_wait<=10:
            s_T_wait = T_wait/10
 

        else:
            s_T_wait=1
        return s_size,s_yita,s_T_wait


    def multiple_rechan(self,action):  # '["size_nor", "T_wait", "T_tr", "prelabel", "relay", "cam", "pic_local", "local", "CRF", "timeslot", "number"])
        #action (n_ca+CRF)：前面有且只有一个1，CRF是0-4
        # 输入图片信息，作用到函数上
        for i in range(0,self.n_ca):

            if action[i] == -1:
                if self.x_num[i] != 0:
                    picture_local = self.X[i * self.T_max, 9]
                    index = int(picture_local)
                    size = self.back[index, 0] * 8#收益要尽量小于回传收益
                    Twait= self.X[i * self.T_max, 3] * 10
                    T_tr = func.transmission(self, size, self.gain)
                    self.R_wait += 0.85 / math.exp(self.bata * (Twait + T_tr + 80))

            if action[i]==1:
                if self.x_num[i]!=0:
                    picture_local=self.X[i*self.T_max,9]
                    index = int(picture_local)
                    self.I[0]=self.back[index,int(0+action[self.n_ca]*2)]*8
                    self.I[1]=self.X[i*self.T_max,3]*10
                    T_tr=func.transmission(self, self.I[0], self.gain)
                    self.I_pointer=math.ceil(T_tr/10)
                    self.I[2]=  T_tr
                    self.I[3] = self.back[index,int(1+action[self.n_ca]*2)]
                    self.I[4] = i
                    self.I[5] = 1
                    self.R_tr += self.I[3]/math.exp(self.bata*(self.I[2]+self.I[1]+80))
                    if self.count==1:
                        self.pic+=1
                        self.trans+=self.I[2]+self.I[1]+80
                        self.acc+=self.I[3]

            if action[i]==0:
                if self.x_num[i]!=0:
                    index = int(i * self.T_max)
                    self.R_not += self.X[index, 5] / math.exp(self.bata*(self.X[index,3]*10+40))
                    if self.count==1:
                        self.pic+=1
                        self.trans+=self.X[index,3]*10+40
                        self.acc+=self.X[index, 5]
                        self.transnot += self.X[index,3]*10+40
                        self.accback += self.X[index, 5]
                        self.picback += 1
             #考虑给等待图片带来一定收益

            if action[i]!=-1:
                if self.x_num[i]!=0:
                    index = int(i*self.T_max)
                    for z in range(0, self.T_max - 1):
                        self.X[index + z, :] = self.X[index + z + 1, :]
                    self.X[index + self.T_max - 1, :] = np.zeros(12)
                    self.x_num[i]-=1

        reward=self.R_tr+self.R_not+self.R_wait
        reward_test = self.R_tr + self.R_not
        # reward = self.R_tr + self.R_not

        self.R_tr=0
        self.R_not=0
        self.R_wait = 0
        return reward,reward_test

    def transmission(self,size, gain):  # 计算传输时间T(ms),信道状况gain(仅瑞利衰落),K为图片大小(KB),rho为压缩率
        Bandwidth = 700e3
        N0 = 10 ** -9.7
        P = 0.03
        R = 30
        alpha = 3.5
        lamda = 1

        Krho = size  # 将byte转换为bit
        c = 0  # 每5ms相干时间内传输的数据量
        tau = 10e-3  # 相干时间5ms
        T = 0  # 传输时间
        while (Krho > 0):
            # gain_5ms_ral = func.jointpdf1(self,self.mean,self.tau, self.gain)
            # print(gain_5ms_ral)
            # gain_5ms = R ** (-1 * alpha) * gain_5ms_ral
            gain_5ms = R ** (-1 * alpha) * self.gain
            c = self.Bandwidth * math.log2(1 + P * gain_5ms / (N0))
            # self.gain=gain_5ms_ral
            # print(P*gain_5ms/(N0))
            SINR = 10 * math.log10(P * gain_5ms / (N0))  # SINR为信干噪比，也可以用10*log（SINR）的dB形式表示
            T = T + 10e-3
            Krho = Krho - c * tau

        T = T + Krho / c
        T = T * 1e3
        # print(T)
        return T

    def atoaction(self,In,a,state,table,picindex):#IN=0，信道空闲


        action = np.zeros(self.n_ca + 1)
        if In!=0:
            In=1
        if a >= 12:
            if (a-60)//4==0:
                action[2]=0
            if (a-60)//4==1:
                action[2]=-1
            if (a-60)%4//2==0:
                action[1]=0
            if (a-60)%4//2==1:
                action[1]=-1
            if (a-60)%2==0:
                action[0]=0
            if (a-60)%2==1:
                action[0]=-1
        else:
            if a % 12 // 4 == 0:
                action[0] = 1 - In             #这里的设置可能有问题
                action[1] = -(a % 12 % 4 // 2)
                action[2] = -(a % 12 % 4 % 2)
            if a % 12 // 4 == 1:
                action[1] = 1 - In
                action[0] = -(a % 12 % 4 // 2)
                action[2] = -(a % 12 % 4 % 2)
            if a % 12 // 4 == 2:
                action[2] = 1 - In
                action[0] = -(a % 12 % 4 // 2)
                action[1] = -(a % 12 % 4 % 2)
        for index in range(0,self.n_ca):
            if action[index]==1:
                action[self.n_ca ]=table[int(picindex[index])]
        return action

    def classi(self,size, yita):
        if size <= 0.1782:
            S = 0
        elif size <= 0.3666:
            S = 1
        elif size <= 0.5669:
            S = 2
        elif size <= 0.7725:
            S = 3
        else:
            S = 4

        if yita <= 0.1117:
            Yita = 0
        elif yita <= 0.1319:
            Yita = 1
        elif yita <= 0.2177:
            Yita = 2
        elif yita <= 0.3141:
            Yita = 3
        else:
            Yita = 4

        return S, Yita

