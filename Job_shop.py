import numpy as np
import random
#from Instance_Generator import Processing_time,A,D,M_num,Op_num,J,O_num,J_num
from Object_for_FJSP import Object
 
class Situation:
    def __init__(self,J_num,M_num,O_num,J,Processing_time,D,Ai,Change_cutter_time, Repair_time, EL):
        self.Ai=Ai                  #arriving time
        self.D=D                    #delivery time
        self.O_num=O_num            #operation num
        self.M_num=M_num            #machine num
        self.J_num=J_num            #job num
        self.J=J                    #operation num of each job
        self.Processing_time = Processing_time   # processing time
        self.CTK=[0 for i in range(M_num)]      #last machine working time
        self.OP=[0 for i in range(J_num)]       #the number of finished operations for each job
        self.UK=[0 for i in range(M_num)]       #machine utilization
        self.CRJ=[0 for i in range(J_num)]      #completion rate of jobs
        # job set£º
        self.Jobs=[]
        for i in range(J_num):
            F=Object(i)
            self.Jobs.append(F)
        #machine set
        self.Machines = []
        for i in range(M_num):
            F = Object(i)
            self.Machines.append(F)

        self.Change_cutter_time=Change_cutter_time
        self.Repair_time=Repair_time
        self.EL = EL
        # ---------------breakdown probability----------
        self.BP = 0.1
 
    def _Update(self,Job,Machine):
        self.CTK[Machine]=max(self.Machines[Machine].End)
        self.OP[Job]+=1
        self.UK[Machine]=sum(self.Machines[Machine].T)/self.CTK[Machine]
        self.CRJ[Job]=self.OP[Job]/self.J[Job]

    def Features(self):
 
        #1 uk
        U_ave=sum(self.UK)/self.M_num
        K=0
        for uk in self.UK:
            K+=np.square(uk-U_ave)
        #2 average uk
        U_std=np.sqrt(K/self.M_num)
        #3 average CRJ
        CRJ_ave=sum(self.CRJ)/self.J_num
        K = 0
        for uk in self.CRJ:
            K += np.square(uk - CRJ_ave)
        #4 CRJ std
        CRJ_std=np.sqrt(K/self.J_num)
        #5 Estimated tardiness rate Tard_e
        T_cur=sum(self.CTK)/self.M_num
        N_tard,N_left=0,0
        for i in range(self.J_num):
            if self.J[i]>self.OP[i]:
                N_left+=self.J[i]-self.OP[i]
                T_left=0
                for j in range(self.OP[i]+1,self.J[i]):
                    M_ij = []
                    for k in range(self.M_num):
                        if self.Processing_time[i][j][k] > 0:
                            PT=self.Processing_time[i][j][k]
                            M_ij.append(PT)
                    T_left+=sum(M_ij)/len(M_ij)
                    if T_left+T_cur>self.D[i]:
                        N_tard+=self.J[i]-j+1
        try:
            Tard_e=N_tard/N_left
        except:
            Tard_e =9999
        #6 Actual tardiness rate Tard_a
        N_tard, N_left = 0, 0
        for i in range(self.J_num):
            if self.J[i] > self.OP[i]:
                N_left += self.J[i] - self.OP[i]
                try:
                    if self.CTK[i] > self.D[i]:
                        N_tard += self.J[i] - j
                except:
                    pass
        try:
            Tard_a = N_tard / N_left
        except:
            Tard_a =9999
        return U_ave,U_std,CRJ_ave,CRJ_std,Tard_e,Tard_a
 
    #Composite dispatching rule 1
    #return Job,Machine
    def rule1(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = UC_Job[np.argmin([(self.OP[i]) / (self.J[i]) for i in UC_Job])]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij =self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # tool change
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij+PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine=np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i,Machine

    # Composite dispatching rule 2
    # return Job,Machine
    def rule2(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = UC_Job[np.argmin([(self.OP[i]) / (self.J[i]) for i in UC_Job])]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i] 
        A_ij = self.Ai[Job_i] 
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(max(C_ij, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 3
    # return Job,Machine
    def rule3(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = UC_Job[np.argmin([(self.OP[i]) / (self.J[i]) for i in UC_Job])]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]  
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        # print('This is from rule 1:',Mk)
        Machine = random.choice(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 4
    # return Job,Machine
    def rule4(self):
        T_cur = sum(self.CTK) / self.M_num
        
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            jobs = []
            for i in UC_Job:
                try:
                    C_i = max(self.Jobs[i].End)
                except:
                    C_i = self.Ai[i]
                jobs.append((C_i + T_cur - self.D[i]) / self.EL[i])
            Job_i = UC_Job[np.argmin(jobs)]
        else:
            T_ijave = []
            for i in Tard_Job:
                T_ijave.append(self.D[i] - T_cur / (3- self.EL[i] + 1))
            Job_i = Tard_Job[np.argmin(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]  
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # tool change
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij + PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 5
    # return Job,Machine
    def rule5(self):
        T_cur = sum(self.CTK) / self.M_num
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            jobs = []
            for i in UC_Job:
                try:
                    C_i = max(self.Jobs[i].End)
                except:
                    C_i = self.Ai[i]
                jobs.append((C_i + T_cur - self.D[i]) / self.EL[i])
            Job_i = UC_Job[np.argmin(jobs)]
        else:
            T_ijave = []
            for i in Tard_Job:
                T_ijave.append(self.D[i] - T_cur / (3 - self.EL[i] + 1))
            Job_i = Tard_Job[np.argmin(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]  
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(max(C_ij, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 6
    # return Job,Machine
    def rule6(self):
        T_cur = sum(self.CTK) / self.M_num
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            jobs = []
            for i in UC_Job:
                try:
                    C_i = max(self.Jobs[i].End)
                except:
                    C_i = self.Ai[i]
                jobs.append((C_i + T_cur - self.D[i]) / self.EL[i])
            Job_i = UC_Job[np.argmin(jobs)]
        else:
            T_ijave = []
            for i in Tard_Job:
                T_ijave.append(self.D[i] - T_cur / (3 - self.EL[i] + 1))
            Job_i = Tard_Job[np.argmin(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]  
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        # print('This is from rule 1:',Mk)
        Machine = random.choice(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 7
    # return Job,Machine
    def rule7(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = random.choice(UC_Job)
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]  
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # tool change
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij + PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 8
    # return Job,Machine
    def rule8(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = random.choice(UC_Job)
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]  
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(max(C_ij, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    # Composite dispatching rule 9
    # return Job,Machine
    def rule9(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = random.choice(UC_Job)
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  
        A_ij = self.Ai[Job_i]  
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        # print('This is from rule 1:',Mk)
        Machine = random.choice(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    def scheduling(self,action):
        Job,Machine=action[0],action[1]
        O_n=len(self.Jobs[Job].End)
        # print(Job, Machine,O_n)
        Idle=self.Machines[Machine].idle_time()
        try:
            last_ot=max(self.Jobs[Job].End)
        except:
            last_ot=0
        try:
            last_mt=max(self.Machines[Machine].End)
        except:
            last_mt=0
        Start_time=max(last_ot,last_mt)
        PT = self.Processing_time[Job][O_n][Machine]
        if self.change_cutter(Job,Machine) == 1:
            PT += self.Change_cutter_time[Machine]
        # machine breakdown
        break_down = random.random()
        uk_bp = np.percentile(self.CTK, 90)
        # machine in the top 10% of machine utilization is more likely to breakdown
        if self.CTK[Machine] >= uk_bp:
            break_down = min(break_down, random.random())
        if break_down < self.BP:
            PT += self.Repair_time[Machine]
        for i in range(len(Idle)):
            if Idle[i][1]-Idle[i][0]>PT:
                if Idle[i][0]>last_ot:
                    Start_time=Idle[i][0]
                    pass
                if Idle[i][0]<last_ot and Idle[i][1]-last_ot>PT:
                    Start_time=last_ot
                    pass
        end_time=Start_time+PT
        self.Machines[Machine]._add(Start_time,end_time,Job,PT)
        self.Jobs[Job]._add(Start_time,end_time,Machine,PT)
        self._Update(Job,Machine)

    def reward1(self,Ta_t,Te_t,Ta_t1,Te_t1):
        '''
               :param Ta_t: Tard_a(t)
               :param Te_t: Tard_e(t)
               :param Ta_t1: Tard_a(t+1)
               :param Te_t1: Tard_e(t+1)
               :return: reward
        '''
        if Ta_t1<Ta_t:
           rt=1
        else:
            if Ta_t1>Ta_t:
                rt=-1
            else:
                if Te_t1<1.*Te_t:
                    rt=1
                else:
                    if Te_t1>Te_t:
                        rt=-1
                    else:
                        rt = 0
        return rt

    def reward2(self,U_t,U_t1):
        '''
               :param U_t: U_ave(t)
               :param U_t1: U_ave(t+1)
               :return: reward
        '''
        if U_t1 > U_t:
            rt = 1
        else:
            if U_t1 > 0.9 * U_t:
                rt = 0
            else:
                rt = -1
        return rt


    # tool change
    def change_cutter(self,Job,Machine):
        assigned_jobs = self.Machines[Machine].assign_for
        assigned_machines = self.Jobs[Job].assign_for
        if (len(assigned_machines) != 0 and assigned_machines[-1] != Machine) or (len(assigned_jobs) != 0 and assigned_jobs[-1] != Job):
            return 1
        return 0
#Sit=Situation(J_num,M_num,O_num,J,Processing_time,D,A)