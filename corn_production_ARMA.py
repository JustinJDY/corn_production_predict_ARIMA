import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm

df=pd.read_csv('cornproduction.csv', encoding='utf-8', index_col='years')
df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引
ts = pd.Series(df['production'],df.index)  # 生成pd.Series对象

def draw_trend(timeseries):
    # 决定起伏统计
    rolmean = timeseries.rolling(window=12,center=False).mean()  # 对size个数据进行移动平均
    rol_weighted_mean = timeseries.ewm(span=12).mean()  # 对size个数据进行加权移动平均
    rolstd = timeseries.rolling(window=12).std()  # 偏离原始值多少
    # 画出起伏统计
    plt.plot(timeseries, color='royalblue', label='Original',linewidth=1)
    plt.plot(rolmean, color='firebrick', label='Rolling Mean',linewidth=1)
    plt.plot(rol_weighted_mean, color='darkgreen', label='weighted Mean',linewidth=1)
    plt.plot(rolstd, color='grey', label='Rolling Std',linewidth=1)
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

def test_stationarity(timeseries):
    # ADF检验
    print('Result of Dickry-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def draw_acf_pacf(ts, lags=31):
    # 自相关和偏自相关
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

# 画出原序列
plt.plot(ts,color='royalblue', linewidth=1)
plt.xlabel("Years")
plt.ylabel("Millions of Bushels")
plt.title('The Original Time Series of Corn Production')
plt.show()

# 调用起伏统计函数
draw_trend(ts)
plt.show()
# ADF检验
test_stationarity(ts)

# 画出对数序列 & 调用起伏统计函数 & ADF检验
ts_log = np.log(ts)
plt.plot(ts_log,color='royalblue', linewidth=1)
plt.xlabel("Years")
plt.ylabel("log")
plt.title('The log Series of Corn Production')
plt.show()
test_stationarity(ts_log)

# 一阶差分 差分后检验
ts_log_diff_1 = ts_log.diff(1)
ts_log_diff_1.dropna(inplace=True)
test_stationarity(ts_log_diff_1)
plt.subplot(211)
plt.plot(ts_log_diff_1)
plt.title('Differential_1')
# 一阶差分后再一阶差分 差分后检验
ts_log_diff_1_1 = ts_log_diff_1.diff(1)
ts_log_diff_1_1.dropna(inplace=True)
test_stationarity(ts_log_diff_1_1)
plt.subplot(212)
plt.plot(ts_log_diff_1_1)
plt.title('Differential_1_1')
plt.show()

# 自相关和偏自相关 确定阶数
draw_acf_pacf(ts_log_diff_1)

# 信息准则定阶
print(sm.tsa.arma_order_select_ic(ts_log_diff_1,max_ar=6,max_ma=4,ic='aic')['aic_min_order'])  # AIC
print(sm.tsa.arma_order_select_ic(ts_log_diff_1,max_ar=6,max_ma=4,ic='bic')['bic_min_order'])  # BIC
print(sm.tsa.arma_order_select_ic(ts_log_diff_1,max_ar=6,max_ma=4,ic='hqic')['hqic_min_order']) # HQIC

# ARMA模型
order = (1,2)
data = ts_log_diff_1
train = data[:-10]
test = data[-10:]
model = ARMA(train,order).fit()
plt.figure(figsize=(15,5))
plt.plot(train,label='real value')
plt.plot(model.fittedvalues,label='fitted value')
plt.legend(loc=0)
plt.show()

# 拟合结果
delta = model.fittedvalues - train
score = 1 - delta.var()/train.var()
print(score)

# 预测
predictions = []
history = train
for t in range(len(test)):
    i = -10+t
    history = data[:i]
    model = ARMA(history, order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    obs = test[t]
    predictions.append(yhat)
    print('predicted={}, expected={}'.format(yhat[0], obs))

plt.figure(figsize=(15,5))
plt.plot(test,label='real value')
plt.plot(predictions,label='fitted value')
plt.legend(loc=0)
plt.show()
