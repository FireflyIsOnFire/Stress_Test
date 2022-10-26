import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from datetime import date
import numpy.random as npr
import matplotlib.pyplot as plt
from pylab import mpl
import scipy.optimize as sco
from scipy import stats
import seaborn as sns
import math

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus']=False
pd.set_option('display.max_columns',None)



begin=date(2018,2,18)
end=date(2020,3,23)  #always get bugged, weird
#print(end)
interval=(end-begin).days
#print(interval)

tickers = ['BTC-USD','ETH-USD','USDT-USD','ADA-USD','XRP-USD','DOGE-USD']

df = pd.DataFrame()
for i in tickers:
    df[i] = web.DataReader(i,'yahoo',begin,date.today())['Adj Close']
df = np.log(df/df.shift(1))
df.to_csv('data')
#print(df)

historical_data_for_opt = df[:365]
stress_data = df[732:766]

historical_data_for_opt.to_csv('historical_data_for_opt')
stress_data.to_csv('stress_data')
#print(historical_data_for_opt,stress_data)
asset_number = len(stress_data.columns)
#print(asset_number)

h_length = len(historical_data_for_opt)
s_lenght = len(stress_data)

return_an_his = historical_data_for_opt.mean()*h_length
cov_an_his = historical_data_for_opt.cov()*h_length
risk_free_rate = np.log(1.04)

'''
port_return =[]
port_vola = []
sharpe_ratio = []
for i in range(100000):
    weight = np.random.random(asset_number)
    weight = weight/np.sum(weight)
    returns = np.dot(weight, return_an_his)
    volatility = np.sqrt(np.dot(weight.T, np.dot(cov_an_his, weight)))
    SPI = (returns - risk_free_rate)/volatility
    port_return.append(returns)
    port_vola.append(volatility)
    sharpe_ratio.append(SPI)
port_vola = np.array(port_vola)
port_return = np.array(port_return)
sharpe_ratio = np.array(sharpe_ratio)


plt.style.use('bmh')
plt.figure(figsize=(8, 4))
plt.scatter(port_vola, port_return, c=sharpe_ratio,cmap='RdYlGn', edgecolors='black',marker='.')
plt.grid(True)
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier of Portfolios')
#plt.show()

def statistics(weights):
    weights = np.array(weights)
    #print("weight测试：",weights)
    #此处我有疑虑，因为我算的是ln的收入，真实收入应该为Preturn x e^Preturn,有机会调整一下
    P_return = np.sum(historical_data_for_opt.mean(numeric_only=True) * weights) * h_length
    P_volatility = np.sqrt(np.dot(weights.T, np.dot(historical_data_for_opt.cov() * h_length, weights)))
    #print("Portfolio return: %Preturn"%Preturn,"Portfolio volatility: %Pvolatility"%Pvolatility) 这里我不明白为什么会导致后面代码错误，只是尝试print信息，奇怪
    return np.array([P_return, P_volatility, ((P_return-risk_free_rate)/ P_volatility)])
   #有点不确定,sharpe ratio用对数回报率还是平均回报率
def min_func_sharpe(weights):
    return -statistics(weights)[2]

bnds = tuple((0, 1) for x in range(asset_number))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(min_func_sharpe, asset_number * [1. / asset_number,], method='SLSQP',  bounds=bnds, constraints=cons)
print(opts['x'].round(3)) #得到各股票权重
print(statistics(opts['x']).round(3)) #得到投资组合预期收益率、预期波动率以及夏普比率
'''

#opt_invest = opts['x'].round(3)*historical_data_for_opt

#stress_data

# assumes the opt. weight is:
opt_weight = [0.4,0.2,0.05,0.15,0.1,0.1]



for i in range(4):
    portfolio = pd.DataFrame()
    price_change = []
    stress_period = len(stress_data)
    steps_length = round(np.sqrt(stress_period))
    rounds = round(stress_period / steps_length)
    print('stress period: ', stress_period, '\n', 'steps: ', steps_length, '\n', 'rounds: ', rounds, '\n')

    record_scenario = []
    record_portfolio = []
    for i in range(rounds):
        rand = np.random.randint(0, stress_period)
        print('Scenario simulated in round: ', (i + 1))
        if rand + steps_length <= stress_period:
            print(stress_data[rand:(rand + steps_length)])
            price_change = stress_data[rand:(rand + steps_length)].sum(axis=0)
            t = stress_data[rand:(rand + steps_length)]
            portfolio = price_change * opt_weight
            # print('\n','price change in this round: ',price_change)
            # t.append(t)
        else:
            # x = stress_data[(rand - stress_period):]
            # y = stress_data[:(steps_length+rand-stress_period)]
            # z = pd.concat(x,y,ignore_index=True)
            print(stress_data[(rand - stress_period):], '\n', stress_data[:(steps_length + rand - stress_period)])
            price_change = stress_data[(rand - stress_period):].sum(axis=0) + stress_data[:(
                        steps_length + rand - stress_period)].sum(axis=0)
            portfolio = price_change * opt_weight
            # print('细节检查：','rand: ',rand,'diyige: ',stress_data[(rand - stress_period):],'\n', stress_data[(steps_length+rand-stress_period)])
            # print('\n','price change in this round: ',price_change)
        record_scenario.append(price_change)
        record_portfolio.append(portfolio)
        # print(t)
    record_scenario = np.array(record_scenario).T
    record_portfolio = np.array(record_portfolio)
    # print('price change in whole scenario: ',record_scenario,'\n','portfolio price change: ',record_portfolio)

    portfolio = record_portfolio.sum(axis=1)
    # print(portfolio,'\n')
    portfolio_value = np.array(portfolio)
    # print('portfolio value: ','\n',portfolio_value,'\n')

    port_in_scenario = []
    # print(len(portfolio_value))
    for i in range(len(portfolio_value)):
        if i == 0:
            portfolio[i] = portfolio_value[i]
            print('i: ', portfolio[i], 'i:', portfolio_value[i])
        else:
            portfolio[i] = portfolio_value[i] + portfolio_value[(i - 1)]
            print('i: ', portfolio_value[i], 'i-1: ', portfolio_value[(i - 1)])
    # print('portfolio log return: ','\n',portfolio,'\n')

    real_portfolio_return = math.e ** (portfolio)
    real_portfolio_return = np.insert(real_portfolio_return, 0, 1)
    print('real portfolio return: ', real_portfolio_return)

    plt.style.use('bmh')
    plt.figure(figsize=(9, 7))
    plt.grid = True
    plt.xlabel('Scenario round', fontsize=15)
    plt.ylabel('Real portfolio return', fontsize=15)
    plt.title('Stress test with historical scenario', fontsize=15)
    plt.plot(real_portfolio_return, color='darkblue')
    # plt.plot()
    plt.show()

