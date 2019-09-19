
# coding: utf-8

# In[1]:


# 实现一个单变量梯度下降的类


# In[8]:


import numpy as np


# In[84]:


class singleRegression:
    def __init__(self, alpha=0.1, maxIterations=5000, jingquedu=0.001):
        '''
        初始化一个单变量线性回归模型
        Params:
            alpha - 步长
            maxIterations - 最大迭代次数
            jingquedu - 满足精度后不再迭代
        '''
        self.alpha=alpha
        self.maxIterations=maxIterations
        self.jd = jingquedu
        self.costs = []
    
    def hyt(self, t1, t2):
        return np.array([x + t1 for x in np.zeros(self.length)]) + np.array([t2 * x for x in self.data[:][0]])
    
    def cost(self, t1, t2):
        '''
        求当前t1, t2 损失函数的值
        '''
        ht = self.hyt(t1, t2)
        j = np.power(([ht - self.data[:][1]]), 2)
        return j.sum()/(2 * len(ht))
        
    def partial(self, isZero, t1, t2):
        '''
        求偏导数, 返回偏导值/m
        '''
        cha = self.hyt(t1, t2) - self.data[:][1]
        if isZero:
            return cha.sum()/self.length
        x1 = self.data[0]
        cha = x1 * cha
        return cha.sum() / self.length
        
    def fit(self, data):
        '''
        使用参数训练模型
        '''
        self.costs = []
        self.data = data
        self.length = len(data[0])
        if data.shape[0] != 2:
            raise Exception("数据类型不正确!")
        t1 = 0
        t2 = 0
        for i in range(self.maxIterations):
            tt1 = t1 - self.alpha * self.partial(True, t1, t2)
            tt2 = t2 - self.alpha * self.partial(False, t1, t2)
            t1 = tt1
            t2 = tt2
            costt = self.cost(t1, t2)
            self.costs.append(costt)
            if costt <= self.jd:
                break
        self.t1 = t1
        self.t2 = t2
        
    def predict(self, data):
        return [x + self.t1 for x in np.zeros(len(data))] + self.t2 * np.array(data)


# In[93]:


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sr = singleRegression()
    d1 = np.array([1, 2, 3, 4, 5])
    d2 = d1 * 3 + 18
    datas = np.array([d1, d2])   # 伪造一个数据
    sr.fit(datas)
    testdata = [1, 3, 5, 7, 9]
    print('testdata = [1, 3, 5, 7, 9]')
    print('预测值为:',sr.predict(testdata))
    print('\n')
    print('损失值如下:')
    plt.plot(np.arange(0, len(sr.costs), 1), sr.costs)
    plt.pause(0)

# In[76]:


# 时间 2019/9/8 13:19

