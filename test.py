'''
test Multi-armed bandits with UCB1 policy and Thompson Sampling
author: An Yan, June 2017
'''
import numpy as np
import argparse
import math
import random
import matplotlib.pyplot as plt
import time

from arm import arm
from UCB import UCB
from posterior import posterior


'''
init and start game for one round
'''
class MAB:
    def __init__(self,p_list,args,alpha=2):
        self.k = 5
        self.T = args.t_rounds
        self.alpha = alpha  # UCB confidence interval parameter
        self.bandts = list()   # init all arms
        self.p_list=p_list
        for p in p_list:
            self.bandts.append(arm(p))
        # the seen rewards won every round.
        self.rewards = list()    #[0]*self.T
        self.N = [0]*self.k  # N_i(t), cumulated # times arm i got pulled
        # counter of # of success and failures of each arm, dictionary
        self.SF_counter = dict(zip(np.arange(self.k),[(0,0)]*self.k))

        self.UCB = [0]*self.k # maintain a list of UCB values of all arms at time t
        self.hat_mu_list = [0]*self.k  # list of currentl round of hat_mu
        self.var_list = list() # maintain a list of beta vars at time t for all arms

        self.mu_best = max(p_list)
        self.best_arm= np.argmax(p_list) 
        self.progress_best_arm = list()  # record   N5,t/t
        self.regrets = list()  #[0]*self.T  # total regrets after each round
        self.N_matrix=np.zeros((self.T,self.k))


    def _start_game(self):
        for t in np.arange(self.T):
            hat_mu_list = list() # for all estimated mus
            var_list = list()
            print("round ------",t)
            for i in np.arange(self.k):
                # draw hat_mu according to Beta(S_i(t)+a, F_i(t)+b)
                a,b = self.SF_counter[i][0]+1, self.SF_counter[i][1]+1
                hat_mu_list.append(posterior(a,b).sample())
                var_list.append(posterior(a,b).get_var())
            self.hat_mu_list = hat_mu_list
            self.var_list=var_list
            # get UCB values of each arm and get max arm index, 
            pulled_arm = int(UCB(t,hat_mu_list,self.N,self.var_list,self.alpha).pull_max_arm())
            print("selected arm---------:",pulled_arm)
            # get reward
            reward = self.bandts[pulled_arm].draw_sample()
            self.rewards.append(reward)
            # get regret
            self.regrets.append(self.get_regret(t))
            # update progress on best arm
            self.get_best_arm_progress(t)
            # alarm when best arm progress N5,t/t is above 0.95
            # if self.progress_best_arm[-1]>0.95:
            #     print("first time above 0.95",t)
            #     break
            # update Success and Failure count
            success,fail = (0,0)
            if reward ==1:
                success = self.SF_counter[pulled_arm][0]+1
                fail = self.SF_counter[pulled_arm][1]
                self.SF_counter[pulled_arm]=(success,fail)
            else:
                fail = self.SF_counter[pulled_arm][1]+1
                success = self.SF_counter[pulled_arm][0]
                self.SF_counter[pulled_arm]=(success,fail)
            # self.SF_counter[pulled_arm]=(success,fail)
            print("self.SF_counter[pulled_arm]=(success,fail)",(success,fail))
            # update N_i,t,N_matrix [T,k]
            
            self.N[pulled_arm]+=1
            self.N_matrix[t,:]=self.N

           

            
        self.plot_regret()
        self.plot_cf()
        self.plot_arm_progress()
            

    '''
    total expected regrets E(R_t)= T * mu_best - E ( sum_T Xt )
    = sum_K  (delta_i) * E( N_i(T) )
    '''
    def get_regret(self,t):
        regret = ((t+1) * self.mu_best - np.sum(self.rewards))/float(t+1)
        return regret

    '''
    return N_5,t / t
    '''
    def get_best_arm_progress(self,t):
        self.progress_best_arm.append(self.N[self.best_arm]/float(t+1))
        


    def plot_regret(self):
        plt.figure(figsize=(8,6))
        x = np.arange(self.T)
        y = self.regrets
        plt.plot(x,y)
        plt.title('average regret VS time')
        plt.xlabel('time')
        plt.ylabel('average regret')
        filename='regret_time_'+str(self.T)+'.png'
        plt.savefig(filename)


    '''
    plot confidence interval at each time t
    '''
    def plot_cf(self):
        x=np.arange(1,self.k+1)
        print(self.var_list)
        
        plt.figure(figsize=(6,4))
        plt.errorbar(x, self.hat_mu_list, yerr=self.var_list,fmt='o',label='estimated mean')
        plt.scatter(x, self.p_list, marker='*',label='true mean',color='g')
        plt.xlim(0, 6)
        title='confidence interval, T='+str(self.T)
        plt.ylabel("confidence interval")
        plt.xlabel('arm')
        plt.title(title)
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        plt.xlim(0, 6)
        filename='conf_intv_'+str(self.T)+'.png'
        plt.savefig(filename,bbox_inches="tight")
        plt.close()

    '''
    for each arm a, make a plot where the x-axis indexes 
    the time t and the y-axis shows Na,t/t
    '''
    def plot_arm_progress(self):
        fig = plt.figure(figsize=(8,6))
        x_range = np.array(range(0,self.T))

        plt.title("Ni_t VS time")
        plt.xlabel("time")
        plt.ylabel("Ni_t")
        # plt.grid()
        for i in range(self.k):
            plt.plot(x_range,self.N_matrix[:,i],'-',label='arm='+str(i+1))
    
        # plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        plt.legend(loc='best')
        filename = 'arm+progress_'+str(self.T)+'.png'
        fig.savefig(filename,bbox_inches="tight")
        plt.close()

       

    def reset_game(self):
        return None



def main():
    args = get_args()
    # a list of p for Berboulli
    p_list = [1/6.,1/2.,2/3.,3/4.,5/6.]
    alpha = 1
    mab = MAB(p_list,args,alpha)

    mab._start_game()
    print("mu_list",mab.hat_mu_list)
    print("final progress",mab.progress_best_arm[-10:])
    print("regrets",mab.regrets[-10:])

    print("first time get to 0.95-------")
    N_list = mab.progress_best_arm
    alarm=0
    for n in range(len(N_list)):
        if N_list[n]>0.95:
            print(n)
            alarm=n
            break

        



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--t_rounds",action = "store",type=int,
    default = 2000, help= "input the number of rounds run")
    return parser.parse_args()


if __name__=='__main__':
    main()
