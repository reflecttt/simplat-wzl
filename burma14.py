import numpy as np
import pandas as pd
import random as rd
import time
import matplotlib.pyplot as plot

np.random.seed(2)  # reproducible
N_STATE=14
ORI = rd.randint(0, N_STATE - 1)
EPSILON_MAX = 0.9
global EPSILON
EPSILON=0
EPSILON_INCRSEMENT = 0.0001
ALPHA = 0.1     # learning rate ok
GAMMA = 0.9    # discount factor
MAX_EPISODES = 20000   # maximum episodes
ABSMAX=9
cities = [[-1, -1] for i in range(N_STATE)]
dis=[[-1]*N_STATE for i in range(N_STATE)]
MIN_BOUND=11.5

def pre():
    path = r'D:\wangzunliang\Desktop\burma.xlsx'
    df = pd.read_excel(path, usecols=[1, 2])
    t_list = df.values.tolist()
    for i in range(N_STATE):
        cities[i][0] = t_list[i][0]
        cities[i][1] = t_list[i][1]
        for j in range(N_STATE):
            if (dis[i][j] == -1):
                temp = np.sqrt((pow(t_list[i][0] - t_list[j][0], 2) + pow(t_list[i][1] - t_list[j][1], 2)) / 10.0)
                dis[i][j] = np.around(temp)
                dis[j][i] = dis[i][j]

def calc_dis(path):
    length=0
    for i in range(len(path)-1):
        length+=dis[path[i]][path[i+1]]
    return length

def buid_q_table():
    table=np.zeros((N_STATE,N_STATE))
    return table

def update_env(path,episode,Q_table):#其实只有当path填满才会调用该函数
    # if(len(path)==N_STATE):
    global EPSILON
    if (EPSILON_MAX > EPSILON):
        EPSILON += EPSILON_INCRSEMENT
    interaction = 'Episode %s:' % (episode+1)
    print('\r{}'.format(interaction), end='')
    print("path:",path)
    print("distance:",calc_dis(path))
    # print("Q_table:",Q_table.astype(np.int))#.astype(np.int)

def choose_action(state,Q_table,now_path):
    global EPSILON
    now_row=Q_table[state]
    all_pos=[i for i in range(N_STATE)]
    unvisited=set(all_pos)-set(now_path)#还未遍历的点集合
    # print("unvisited:", unvisited)
    if(np.random.uniform()>EPSILON or sum(now_row)==0):
        # print("!!")
        action=np.random.choice(np.array(list(set(all_pos)-set(now_path))))
    else:
        max_value=-10000
        for item in unvisited:
            q_value=now_row[item]
            if(q_value>max_value):
                max_value=q_value
                action=item
    return action

def get_env_feedback(s,a):
    reward=ABSMAX-dis[s][a]#当前f(s,a)=a，也就是s'=a
    return a,reward

def get_best_path(Q_table,begin,flag):
    if(flag):
        print("SSSS Q_table:",Q_table.astype(np.int))
        print("begin:",begin)
    path=[begin]
    all_pos=[i for i in range(N_STATE)]
    for i in range(N_STATE-1):#0-13
        now_row=Q_table[path[i]]
        unvisited=set(all_pos)-set(path)#还未遍历的点集合
        if (flag):
            print("path:", path)
            print("unvisited:",unvisited)
        max_value=-1000
        best_pos=-1
        for item in unvisited:
            value=now_row[item]
            if(value>max_value):
                max_value=value
                best_pos=item
        path.append(best_pos)
    path.append(begin)
    if (flag):
        print("path:",path)
    length=0
    for i in range(len(path)-1):
        if (flag):
            print("path[i]:",path[i],"---path[i+1]:",path[i+1])
            print("dis",dis[path[i]][path[i+1]])
        length+=dis[path[i]][path[i+1]]
    return length

def Q_method():
    Q_table=buid_q_table()
    score=[]
    global EPSILON
    for episode in range(MAX_EPISODES):
        # ori=rd.randint(0,N_STATE-1)
        now_path=[ORI]#本轮路径
        # step_counter=0
        print("epsilon:",EPSILON)
        for i in range(N_STATE-1):
            state = now_path[i]
            # print("state:",state,"before_Q_table[state]:",Q_table[state].astype(np.int))
            action=choose_action(state,Q_table,now_path)
            # print("action:", action)
            state_,reward=get_env_feedback(state,action)
            q_predict=Q_table[state][action]
            if(i!=N_STATE-2):
                q_target=reward+GAMMA*max(Q_table[state_])
            else:#最后一个点 N_STATE-2
                # print("last_state:",state," path:",now_path)#13
                q_target=reward
            Q_table[state][action]+=ALPHA*(q_target-q_predict)
            # print("after_Q_table[state]:",Q_table[state].astype(np.int))
            now_path.append(action)
            # step_counter += 1
        #最后绕回去
        fin_state=now_path[N_STATE-1]
        # print("fin_state:",fin_state)
        # print("now_path:",now_path)
        # action=ori
        # reward=ABSMAX-dis[fin_state][ori]
        action=ORI
        reward=ABSMAX-dis[fin_state][ORI]
        q_predict = Q_table[fin_state][action]
        q_target=reward
        Q_table[fin_state][action] += ALPHA * (q_target - q_predict)
        # now_path.append(ori)
        now_path.append(ORI)
        # step_counter+=1
        # update_env(now_path, episode, step_counter, Q_table)
        update_env(now_path, episode, Q_table)
        now_res=get_best_path(Q_table,ORI,False)
        score.append(now_res)
        # if(now_res<=MIN_BOUND):
        #     print("提前结束！")
        #     break
    return Q_table,score

def draw(score):
    x=[i for i in range(len(score))]
    plot.plot(x,score,'b-')
    plot.show()

if __name__ == '__main__':
    pre()
    Q_table,score=Q_method()
    draw(score)

