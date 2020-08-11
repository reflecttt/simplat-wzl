import numpy as np
epsilon = 0.5
gamma = 0.98
lr = 0.1
distance =np.array([[0,7,6,1,3],[7,0,3,7,8],[6,3,0,12,11],[1,7,12,0,2],[3,8,11,2,0]])
R_table = 11-distance
space = [0,1,2,3,4]
Q_table = np.zeros((5,5))
for i in range(100):#总共训练10次
    path = [0]#不一定每次都要从0点开始 path=[1,8,22,41,13,....]
    for j in range(4):#每一轮，跑完完整的路径(共5条，还剩4条待遍历)
        s = path[j]#当前路径下的第j个点,path[1]=8
        s_row = Q_table[s]#第8行Q值
        remain = set(space)-set(path)#其他待访问的点的集合
        print("remain:",remain)
        maxvalue = -1000
        for rm in remain:
            Q = Q_table[s, rm]#找到具体的Q值
            if Q>maxvalue:#从剩下几个点中找最优点
                maxvalue = Q
                a = rm
        # print(maxvalue," ",a)
        if np.random.uniform()<epsilon:
            a = np.random.choice(np.array(list(set(space)-set(path))))
        # print(">>>",a)
        if j!=3:#还没到最后结束
            Q_table[s,a] = (1-lr)*Q_table[s,a]+lr*(R_table[s,a]+gamma*maxvalue)
            # print("!!!!!",Q_table[s,a])
        else:
            Q_table[s,a] = (1-lr)*Q_table[s,a]+lr*R_table[s,a]
        path.append(a)
    # TODO 最后需要绕回来
    Q_table[a,0] = (1-lr)*Q_table[a,0]+lr*R_table[a,0]
    path.append(0)
    print("Q_table",Q_table)
    print("path",path)
        #即可得到最佳的TSP路径的Q表