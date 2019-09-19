
# coding: utf-8

# In[12]:


import numpy as np
# logistic 模型, 解决二分类问题
# 由于梯度类似回归, 因此实际上只需要对h(θ)做sigmoid处理即可


# In[24]:


class logisticRegression:
    def __init__(self, alpha=0.001, maxIterations=5000, jd=0.001, mul=1):
        '''
        初始化一个GDregression
        Params:
            # feathreal_nums - 特征数 
            alpha - 学习速率
            maxIterations - 最大迭代次数
            jd - 需求精确度
            mul - 多项式, 待定功能
        '''
        # self.featureal_nums=featureal_nums
        self.alpha = alpha
        self.maxIterations = maxIterations
        self.jd = jd
        self.mul = mul
        self._costs = []
        
    def _sigmoid(self, inputs):
        # sigmoid函数 即 h(\theta)
        # 适用于 ndarray类型
        # print(inputs)
        # print('---')
        return 1.0 / (1.0 + np.exp(-inputs))
    #def _sigmoid(self, x):
        #if x >= 0:
           # return 1.0/(1 + np.exp(-x))
       # else:
           # return np.exp(x)/(1 + np.exp(x))

    def _hypothesis(self, ts, data):
        '''
        使用参数列表来计算该参数对数据的预测结果
        Params:
            ts - 参数, 为2维数组但是每一行只能有一个值, 即类似行向量的转置, 不过ndarray 单行向量转置不会转置为二维\
            需要声明的时候为[[x1, x2, ...xn]]才能够进行转置
        '''
        # 使用添加了 值为1 的列 的数据集
        ret = np.dot(data, ts)
        ret = self._sigmoid(ret)
        return ret   # 返回 一列 ndarray 数组 二维数组
        
    def _fill_1(self, data):
        # 返回添加一列 1 后的特征
        return np.concatenate([np.ones(data.shape[0]).reshape(data.shape[0], 1), data], axis=1)
    
    def _cost(self,ts):
        t = self._hypothesis(ts, self._data_fill_1)
        t = t.reshape(t.shape[0],)
        return np.power((t - self._features), 2).sum()/2/self._data_nums
        #return np.power((self._hypothesis(t1, t2) - self._features), 2).sum()/2/self._data_nums
    
    def fit(self, data, features):
        """
            Params:
                data - 训练集, 需要为二维ndarray
                features - 标签, 需要为一维ndarray, 且为0,1值
        """
        self._data = data
        # 下面这条语句需要先判断data是二维数组才行
        self._x, self._y = data.shape
        self._feathres_nums = len(data[0])    # 获取特征数目    
        self._data_nums = data.shape[0]      # 获得数据条数
        self._data_fill_1 = self._fill_1(data)     # 补充一列1
        self._features = features 
        self._ts = np.zeros((self._feathres_nums + 1, 1))   # 初始化参数数组全为0 得到一个二维数组表示列数组
        self._ts += 0.5    # 二分类不能初始化为0
        # print(self._ts)
        self._costs.clear()
        ts = self._ts
        for i in range(self.maxIterations):
        #for i in range(2): 小循环测试用
            tp = self._hypothesis(ts, data=self._data_fill_1)    # 得到预测结果向量
            tp = tp.reshape(tp.shape[0], )    # 将单列二维向量转变为行向量好与特征做运算
            
            # 以下为对应偏导公式
            ch = tp - self._features        # 预测结果与特征的差向量
            ch.reshape(ch.shape[0], 1)
            dot = np.dot(self._data_fill_1.T, ch)    # 二维(data_nums, features)数据, 得到参数加1行
            sumof = self.alpha * dot
            sumof = sumof.reshape(self._feathres_nums+1, 1) / self._feathres_nums  # 转换为列向量 与theta做运算
            
            tts = ts - sumof 
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
        data = self._fill_1(data)    # 填充 "1" 列
        # ret = np.dot(data, self._ts)
        # ret = np.array([sum(x) for x in ret])
        return self._hypothesis(ts=self._ts, data=data)   # 返回 一列 ndarray 数组


# In[ ]:


# 2019年9月19日18:31:53


# In[2]:


from sklearn.datasets import load_breast_cancer    # 引入外部数据集进行测试


# In[3]:


cancers = load_breast_cancer()
datas = cancers.data
features = cancers.target

from sklearn.preprocessing import StandardScaler
std = StandardScaler()    # 特征缩放
data2 = std.fit_transform(datas)

from sklearn.model_selection import train_test_split
data1, data2, lab1, lab2 = train_test_split(data2, features)    # 分割


# In[10]:


logr = logisticRegression()
logr.fit(data1, lab1)    # 训练


# In[96]:


pre = logr.predict(data2)    # 预测
for i in range(len(pre)):    # 需要手动对预测大于0.5的分成1
    if pre[i] > 0.5:
        pre[i]=1
    else:
        pre[i]=0


# In[97]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score    # 测试方法集


# In[98]:


acu = accuracy_score(lab2, pre)
prec = precision_score(lab2, pre)
rec = recall_score(lab2, pre)
f1 = f1_score(lab2, pre)
cohen = cohen_kappa_score(lab2, pre)


# In[99]:


print('准确率为: ', acu)
print('精确率为: ', prec)
print('召回率为: ', rec)
print('f1率为: ', f1)
print('cohen系数为: ', cohen)


# In[100]:


# 准确率, 精确率, 召回率, f1, cohen系数

