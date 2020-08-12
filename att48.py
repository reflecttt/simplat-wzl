import numpy as np
import pandas as pd
import random as rd
import time
import matplotlib.pyplot as plot

np.random.seed(2)  # reproducible
N_STATE=48
# ORI = rd.randint(0, N_STATE - 1)
EPSILON_MAX = 0.9
global EPSILON
EPSILON=0.0
EPSILON_INCRSEMENT = 0.0001
ALPHA = 0.3     # learning rate 0.1
GAMMA = 0.95    # discount factor 0.9
MAX_EPISODES = 20000   # maximum episodes
ABSMAX=3000
cities = [[-1, -1] for i in range(N_STATE)]
dis=[[-1]*N_STATE for i in range(N_STATE)]
MIN_BOUND=12000

def pre():
    path = r'D:\wangzunliang\Desktop\att48.xlsx'
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

def update_env(episode):#其实只有当path填满才会调用该函数
    global EPSILON
    if (EPSILON_MAX > EPSILON):
        EPSILON += EPSILON_INCRSEMENT
    interaction = 'Episode %s:' % (episode+1)
    print('\r{}'.format(interaction), end='')

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
    reward=reward/100.0
    # print("reward:",reward)
    return a,reward

# def get_best_path(Q_table,begin):
#     path=[begin]
#     all_pos=[i for i in range(N_STATE)]
#     for i in range(N_STATE-1):#0-13
#         now_row=Q_table[path[i]]
#         unvisited=set(all_pos)-set(path)#还未遍历的点集合
#         # print("path:", path)
#         # print("unvisited:",unvisited)
#         max_value=-1000
#         best_pos=-1
#         for item in unvisited:
#             value=now_row[item]
#             if(value>max_value):
#                 max_value=value
#                 best_pos=item
#         path.append(best_pos)
#     path.append(begin)
#     print("best_path:",path)
#     length=0
#     for i in range(len(path)-1):
#         # print("path[i]:",path[i],"---path[i+1]:",path[i+1])
#         # print("dis",dis[path[i]][path[i+1]])
#         length+=dis[path[i]][path[i+1]]
#     print("dis:",length)
#     return length

def get_best_path(Q_table):
    all_path=[]
    all_length=[]
    for begin in range(N_STATE):
        path=[begin]
        all_pos=[i for i in range(N_STATE)]
        for i in range(N_STATE-1):#0-12
            now_row=Q_table[path[i]]
            unvisited=set(all_pos)-set(path)#还未遍历的点集合
            max_value=-1000
            best_pos=-1
            for item in unvisited:
                value=now_row[item]
                if(value>max_value):
                    max_value=value
                    best_pos=item
            path.append(best_pos)
        path.append(begin)
        length=0
        for i in range(len(path)-1):
            length+=dis[path[i]][path[i+1]]
        all_path.append(path)
        all_length.append(length)
        # print("当前path:",path)
        # print("当前length:",length)
    best_length=min(all_length)
    best_path=all_path[all_length.index(best_length)]
    # print("最优path:",best_path)
    print("最优length:",best_length)
    return best_length

def Q_method():
    Q_table=buid_q_table()
    score=[]
    global EPSILON
    for episode in range(MAX_EPISODES):
        ori=rd.randint(0,N_STATE-1)
        now_path=[ori]#本轮路径
        # print("epsilon:",EPSILON)
        for i in range(N_STATE-1):
            state = now_path[i]
            # print("state:",state,"before_Q_table[state]:",Q_table[state].astype(np.int))
            action=choose_action(state,Q_table,now_path)
            # print("action:", action)
            #TODO ①
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
        #最后绕回去
        fin_state=now_path[N_STATE-1]
        # print("fin_state:",fin_state)
        # print("now_path:",now_path)
        action=ori
        reward=ABSMAX-dis[fin_state][ori]
        reward=reward/100.0
        q_predict = Q_table[fin_state][action]
        q_target=reward
        Q_table[fin_state][ori] += ALPHA * (q_target - q_predict)
        # now_path.append(ori)
        update_env(episode) #对于epsilon的更新包括在了learn里
        # now_res=get_best_path(Q_table,ori)
        #TODO ②
        now_res=get_best_path(Q_table)
        score.append(now_res)
        # if(now_res<=MIN_BOUND):
        #     print("提前结束！")
        #     break
    return Q_table,score

def draw(score):
    x=[i for i in range(len(score))]
    plot.plot(x,score,'b-')
    plot.grid()
    plot.show()

if __name__ == '__main__':
    pre()
    Q_table,score=Q_method()
    for item in Q_table:
        print(item)
    draw(score)

