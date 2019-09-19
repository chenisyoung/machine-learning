
# coding: utf-8

# In[1]:


# 实现一个多变量
import numpy as np


# In[264]:


class GDregression:
    def __init__(self, alpha=0.001, maxIterations=5000, jd=0.001, mul=1):
        '''
        初始化一个GDregression
        Params:
            # feathreal_nums - 特征数 
            alpha - 学习速率
            maxIterations - 最大迭代次数
            jd - 需求精确度
            mul - 特征平方数  有空再弄, 比较麻烦
        '''
        # self.featureal_nums=featureal_nums
        self.alpha = alpha
        self.maxIterations = maxIterations
        self.jd = jd
        self.mul = mul
        self._costs = []
        
    def _hypothesis(self,ts):
        ret = np.dot(self._data_fill_1, ts)
        #ret = np.array([sum(x) for x in ret])
        # ret = ret.reshape(ret.shape[0])
        #print('ret == ')
        #print(ret)
        #print('ret.shape', ret.shape)
        return ret   # 返回 一列 ndarray 数组 二维数组
        
    def _fill_1(self, data):
        # 返回添加一列 1 后的特征
        return np.concatenate([np.ones(data.shape[0]).reshape(data.shape[0], 1), data], axis=1)
    
    def _cost(self,ts):
        t = self._hypothesis(ts)
        t = t.reshape(t.shape[0],)
        return np.power((t - self._features), 2).sum()/2/self._data_nums
        #return np.power((self._hypothesis(t1, t2) - self._features), 2).sum()/2/self._data_nums
    
    def fit(self, data, features):
        self._data = data
        # 下面这条语句需要先判断data是二维数组才行
        self._x, self._y = data.shape
        self._feathres_nums = len(data[0])    # 获取特征数目
        self._data_nums = data.shape[0]      # 获得数据条数
        self._data_fill_1 = self._fill_1(data)     # 补充一列1
        self._features = features 
        self._ts = np.zeros(self._feathres_nums + 1).reshape(self._feathres_nums + 1, 1)   # 初始化参数数组全为0 得到一个二维数组表示列数组
        # print(self._ts)
        self._costs.clear()
        ts = self._ts
        for i in range(self.maxIterations):
        #for i in range(2):
            tp = self._hypothesis(ts)
            tp = tp.reshape(tp.shape[0], )
            ch = tp - self._features
            dot = np.dot(self._data_fill_1.T, ch)
            sumof = self.alpha * dot
            #tem = []
            #for item in sumof:
             #   tem.append(sum(item))
            #sumof = np.array(tem)
            sumof = sumof.reshape(self._feathres_nums+1, 1)
            tts = ts - sumof / self._feathres_nums
            # tts = tts.reshape(tts.shape[0], 1)  # 重新变为二维数组
            ts = tts
            cc = self._cost(ts)
            self._ts = ts
            if len(self._costs) == 0:
                pass
            else:
                if cc == self._costs[-1]:
                    break
            self._costs.append(cc)    # 存入损失值
            if cc <= self.jd:
                break
        
    
    def predict(self, data):
        data = self._fill_1(data)
        ret = np.dot(data, self._ts)
        ret = np.array([sum(x) for x in ret])
        # print(ret)
        return ret   # 返回 一列 ndarray 数组


# In[265]:


import matplotlib.pyplot as plt
sr = GDregression()
d1 = np.array([[1], [2], [3], [4], [5]])
d2 = np.array([1, 2, 3, 4, 5]) * 3 + 18
 # 伪造一个数据
sr.fit(d1, d2)
testdata = np.array([[1], [3], [5]])
print('testdata = [[1], [3], [5]]')
print('预测值为:',sr.predict(testdata))
print('\n')
print('损失值如下:')
plt.plot(np.arange(0, len(sr._costs), 1), sr._costs)


# In[269]:


d1 = np.array([1, 2])
d2 = np.array([2, 2])
d3 = np.array([4, 8])
d4 = np.array([2, 1])
data = np.array([d1, d2, d3, d4])
fea = np.array([9, 11, 33, 8])   # 制作一个 y = 2x_0 + 3x_1 + 1 的数据集


# In[279]:


s2 = GDregression(jd=0.01)
s2.fit(data, fea)


# In[280]:


sr._ts    # 哈哈hhhh效果还行
print('损失值如下:')
plt.plot(np.arange(0, len(sr._costs), 1), sr._costs)


# In[281]:


len(s2._costs)


# In[283]:


s2._ts  # 还是有出入 但是很不错了hhh


# In[ ]:


# 没有做特征缩放

