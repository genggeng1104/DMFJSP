import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from collections import deque
from keras import layers,models
from Job_shop import Situation
from keras.optimizers import Adam
import threading
import multiprocessing
 
 
class DQN:
    def __init__(self,):
        self.Hid_Size1 = 10
        self.Hid_Size2 = 50
 
        # ------------Hidden layer=3  10 nodes each layer--------------
        model = models.Sequential()
        model.add(layers.Input(shape=(6,)))
        model.add(layers.Dense(self.Hid_Size1, name='l1'))
        model.add(layers.Dense(self.Hid_Size1, name='l2'))
        model.add(layers.Dense(self.Hid_Size1, name='l3'))
        model.add(layers.Dense(2, name='l4'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        # ------------Hidden layer=7 50 nodes each layer--------------
        model1 = models.Sequential()
        model1.add(layers.Input(shape=(7,)))
        model1.add(layers.Dense(self.Hid_Size2, name='l1'))
        model1.add(layers.Dense(self.Hid_Size2, name='l2'))
        model1.add(layers.Dense(self.Hid_Size2, name='l3'))
        model1.add(layers.Dense(self.Hid_Size2, name='l4'))
        model1.add(layers.Dense(self.Hid_Size2, name='l5'))
        model1.add(layers.Dense(self.Hid_Size2, name='l6'))
        model1.add(layers.Dense(self.Hid_Size2, name='l7'))
        model1.add(layers.Dense(9, name='l8'))
        model1.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        # # model.summary()
        self.model = model
        self.model1 = model1
 
        #------------Q-network Parameters-------------
        self.act_dim=[1,2,3,4,5,6,7,8,9]                        #output
        self.obs_n=[0,0,0,0,0,0]                    #input
        self.gama = 0.95  # γ
        self.global_step = 0
        self.update_target_steps = 200  # update step: C
        self.target_model = self.model
        self.target_model1 = self.model1

        # -------------------Agent-------------------
        self.e_greedy = 0.6
        self.e_greedy_decrement = 0.0001
        self.L = 1  # Number of training episodes L = 20

        # ---------------Replay Buffer---------------
        self.buffer = deque(maxlen=2000)
        self.Batch_size = 16  # Batch Size of Samples to perform gradient descent
 
    def replace_target(self):
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())
        self.target_model.get_layer(name='l4').set_weights(self.model.get_layer(name='l4').get_weights())

        self.target_model1.get_layer(name='l1').set_weights(self.model1.get_layer(name='l1').get_weights())
        self.target_model1.get_layer(name='l2').set_weights(self.model1.get_layer(name='l2').get_weights())
        self.target_model1.get_layer(name='l3').set_weights(self.model1.get_layer(name='l3').get_weights())
        self.target_model1.get_layer(name='l4').set_weights(self.model1.get_layer(name='l4').get_weights())
        self.target_model1.get_layer(name='l5').set_weights(self.model1.get_layer(name='l5').get_weights())
        self.target_model1.get_layer(name='l6').set_weights(self.model1.get_layer(name='l6').get_weights())
        self.target_model1.get_layer(name='l7').set_weights(self.model1.get_layer(name='l7').get_weights())
        self.target_model1.get_layer(name='l8').set_weights(self.model1.get_layer(name='l8').get_weights())
 
    def replay(self):
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()
        # replay the history and train the model
        minibatch = random.sample(self.buffer, self.Batch_size)
        for state, action, reward, next_state, reward_id, done in minibatch:
            target = reward
            target1 = reward
            if not done:
                output = self.target_model.predict(next_state)
                k = np.max(output)
                target = (reward + self.gama * np.argmax(output))
                next_state1 = np.expand_dims(np.append(next_state[0],k), 0)
                target1 = (reward + self.gama *
                          np.argmax(self.target_model1.predict(next_state1)))
            target_f = self.model.predict(state)
            k = np.max(target_f)
            state1 = np.expand_dims(np.append(state[0],k),0)
            target_f1 = self.model1.predict(state1)
            target_f[0][reward_id] = target
            target_f1[0][action] = target1
            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.model1.fit(state1, target_f1, epochs=1, verbose=0)
        self.global_step += 1
 
    def Select_action(self,obs):
        # obs=np.expand_dims(obs,0)
        if random.random()<self.e_greedy:
            rt=random.randint(0,1)
            act=random.randint(0,8)
        else:
            output=self.model.predict(obs)
            rt=np.argmax(output)
            input = np.expand_dims(np.append(obs[0], np.argmax(output)),0)
            act=np.argmax(self.model1.predict(input))
        self.e_greedy = max(
            0.01, self.e_greedy - self.e_greedy_decrement)
        return act,rt
 
    def _append(self, exp):
        self.buffer.append(exp)

    def Instance_Generator(self, M_num, E_ave, New_insert):
        '''
        :param M_num: Machine Number
        :param Initial_job: initial job number
        :param E_ave
        :return: Processing time,A:New Job arrive time,
                                    D:Deliver time,
                                    M_num: Machine Number,
                                    Op_num: Operation Number,
                                    J_num:Job NUMBER
                                    EL:ergency level of each job
        '''
        E_ave = E_ave
        Initial_Job_num = 5
        Op_num = [random.randint(1, 20) for i in range(New_insert + Initial_Job_num)]
        Processing_time = []
        for i in range(Initial_Job_num + New_insert):
            Job_i = []
            for j in range(Op_num[i]):
                k = random.randint(1, M_num - 2)
                T = list(range(M_num))
                random.shuffle(T)
                T = T[0:k + 1]
                O_i = list(np.ones(M_num) * (-1))
                for M_i in range(len(O_i)):
                    if M_i in T:
                        O_i[M_i] = random.randint(1, 50)
                Job_i.append(O_i)
            Processing_time.append(Job_i)
        A1 = [0 for i in range(Initial_Job_num)]
        A = np.random.exponential(E_ave, size=New_insert)
        A = [int(A[i]) for i in range(len(A))]  # New Insert Job arrive time
        A1.extend(A)
        EL = [random.randint(1,3) for i in range(len(A1))]
        T_ijave = []
        for i in range(Initial_Job_num + New_insert):
            Tad = []
            for j in range(Op_num[i]):
                T_ijk = [k for k in Processing_time[i][j] if k != -1]
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(sum(Tad))
        D1 = [int((0.2 + 0.5 * EL[i]) * T_ijave[i]) for i in range(Initial_Job_num)]
        D = [int(A1[i] + (0.2 + 0.5 * EL[i]) * T_ijave[i]) for i in range(Initial_Job_num, Initial_Job_num + New_insert)]
        D1.extend(D)
        O_num = sum(Op_num)
        J = dict(enumerate(Op_num))
        J_num = Initial_Job_num + New_insert

        Change_cutter_time = list(np.zeros(M_num))
        Repair_time = list(np.zeros(M_num))
        for i in range(M_num):
            Change_cutter_time[i] = random.randint(1, 50)
            Repair_time[i] = random.randint(1, 99)

        return Processing_time, A1, D1, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL

    def main(self,J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL):
        k = 0
        x=[]
        Total_tard=[]
        Total_makespan=[]
        Total_uk_ave=[]
        TR=[]
        for i in range(self.L):
            Total_reward = 0
            x.append(i+1)
            print('-----------------------start',i+1,'training------------------------------')
            obs=[0 for i in range(6)]
            obs = np.expand_dims(obs, 0)
            done=False
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL)
            for i in range(O_num):
                k+=1
                # print(obs)
                at,rt=self.Select_action(obs)
                # print(at)

                if at==0:
                    at_trans=Sit.rule1()
                if at==1:
                    at_trans=Sit.rule2()
                if at==2:
                    at_trans=Sit.rule3()
                if at==3:
                    at_trans=Sit.rule4()
                if at==4:
                    at_trans=Sit.rule5()
                if at==5:
                    at_trans=Sit.rule6()
                if at==6:
                     at_trans=Sit.rule7()
                if at == 7:
                    at_trans = Sit.rule8()
                if at == 8:
                    at_trans = Sit.rule9()
                # at_trans=self.act[at]
                print('The', i, 'th operation>>', 'select action:', at, ' ', 'job ', at_trans[0], 'is assigned for machine ', at_trans[1])
                Sit.scheduling(at_trans)
                obs_t=Sit.Features()

                if i==O_num-1:
                    done=True
                #obs = obs_t
                obs_t = np.expand_dims(obs_t, 0)
                # obs = np.expand_dims(obs, 0)
                # print(obs,obs_t)
                if 0 == rt:
                    r_t = Sit.reward1(obs[0][5], obs[0][4], obs_t[0][5], obs_t[0][4])
                else:
                    r_t = Sit.reward2(obs[0][0], obs_t[0][0])
                self._append((obs,at,r_t,obs_t,rt,done))
                if k>self.Batch_size:
                    # batch_obs, batch_action, batch_reward, batch_next_obs,done= self.sample()
                    self.replay()
                Total_reward+=r_t
                obs=obs_t
            total_tadiness=0
            makespan=0
            uk_ave=sum(Sit.UK)/M_num
            Job=Sit.Jobs
            E=0
            K=[i for i in range(len(Job))]
            End=[]
            for Ji in range(len(Job)):
                endTime=max(Job[Ji].End)
                makespan=max(makespan,endTime)
                End.append(endTime)
                if max(Job[Ji].End)>D[Ji]:
                    total_tadiness+=abs(max(Job[Ji].End)-D[Ji])
            print('<<<<<<<<<-----------------total_tardiness:',total_tadiness,'------------------->>>>>>>>>>')
            Total_tard.append(total_tadiness)
            print('<<<<<<<<<-----------------uk_ave:', uk_ave, '------------------->>>>>>>>>>')
            Total_uk_ave.append(uk_ave)
            print('<<<<<<<<<-----------------makespan:', makespan, '------------------->>>>>>>>>>')
            Total_makespan.append(makespan)
            print('<<<<<<<<<-----------------reward:',Total_reward,'------------------->>>>>>>>>>')
            TR.append(Total_reward)
            # plt.plot(K,End,color='y')
            # plt.plot(K,D,color='r')
            # plt.show()
        # plt.plot(x,Total_tard)
        # plt.show()
        return Total_tard,Total_uk_ave,Total_makespan


def train(machine, e_ave, job_insert):
    d = DQN()
    Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = d.Instance_Generator(machine, e_ave, job_insert)
    Total_tard,Total_uk_ave,Total_makespan = d.main(J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL)
    tard_ave=sum(Total_tard)/d.L
    uk_ave=sum(Total_uk_ave)/d.L
    makespan_ave=sum(Total_makespan)/d.L
    std1=0
    std2=0
    std3=0
    for ta in Total_tard:
        std1 += np.square(ta - tard_ave)
    for ua in Total_uk_ave:
        std2 += np.square(ua - uk_ave)
    for ma in Total_makespan:
        std3 += np.square(ma - makespan_ave)
    # std
    std1 = np.sqrt(std1 / d.L)
    std2 = np.sqrt(std2 / d.L)
    std3 = np.sqrt(std3 / d.L)
    print(str(tard_ave)+","+str(std1)+","+str(uk_ave)+","+str(std2)+","+str(makespan_ave)+","+str(std3))

    file = open('/results/' + str(machine) + '-' + str(e_ave) + '-' + str(job_insert), 'w+', encoding='utf-8')
    file.write(str(tard_ave)+","+str(std1)+","+str(uk_ave)+","+str(std2)+","+str(makespan_ave)+","+str(std3))
    file.write("\n")
    file.flush()
    file.close()

def call_back(v):
    print('----> callback pid:', os.getpid(),',tid:',threading.currentThread().ident,',v:',v)

if __name__ == '__main__':
    Total_Machine=[8,12,16]
    Job_insert=[20,30,40]
    E_ave=[50,100,200]
    train(8,50,20)
    # pool = multiprocessing.Pool(27)
    # results = [pool.apply_async(train, args=(machine, e_ave, job_insert, ), callback=call_back) for e_ave in E_ave for machine in Total_Machine for job_insert in Job_insert]
    # pool.close()
    # pool.join()