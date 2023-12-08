# import needed modules
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

######## 期間設定 #######
start = datetime.date(2018, 4, 1)
end = datetime.date(2023, 3, 31)

######## 銘柄 ##########
selected = ['1803.T', '1812.T', '1802.T']

######## 組み合わせ数#######
num_portfolios = 100000

########################
data = yf.download(selected, start=start, end=end)["Adj Close"]
data = data.reindex(columns=selected)

# calculate daily and annual returns of the stocks
returns_daily = data.pct_change()
returns_annual = returns_daily.mean() * 250

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)
# num_portfolios = 200000

# set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter, symbol in enumerate(selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + \
    [stock+' Weight' for stock in selected]

# reorder dataframe columns
df = df[column_order]
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

## グラフパラメータ設定 ###
plt.rcParams.update(plt.rcParamsDefault)
# plt.style.use('dark_background')
# plt.style.use('seaborn-pastel')
# plt.style.use('seaborn-bright')
plt.rcParams["axes.labelcolor"] = 'white'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['xtick.color'] = 'gray'
plt.rcParams['ytick.color'] = 'gray'
plt.rcParams['text.color'] = 'white'

## グラフ領域設定 ###
fig = plt.figure(facecolor='white', figsize=(14, 6), tight_layout="True")
spec = gridspec.GridSpec(ncols=3, nrows=2, height_ratios=[
                         1, 1], width_ratios=[15, 3, 4])

## グラフに領域を割り当て###
ax1 = fig.add_subplot(spec[:, 0], title='Efficient Frontier  ( ' +
                      str(start.year) + " - "+str(end.year) + " )")
ax2 = fig.add_subplot(spec[0, 1], title='Sharpe_port')
ax3 = fig.add_subplot(spec[1, 1], title='Min_variance_port')
ax4 = fig.add_subplot(spec[0, 2], title='Sharpe_port')
ax5 = fig.add_subplot(spec[1, 2], title='Min_variance_port')
fig.patch.set_facecolor('black')

# plot frontier, max sharpe & min Volatility values with a scatterplot
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='jet', edgecolors='black', alpha=1, ax=ax1)
ax1.scatter(x=sharpe_portfolio['Volatility'],
            y=sharpe_portfolio['Returns'], c='red', marker='D', s=50)
ax1.scatter(x=min_variance_port['Volatility'],
            y=min_variance_port['Returns'], c='blue', marker='D', s=50)
for counter, symbol in enumerate(selected):

    #   portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]
    stock = symbol
    max_ticker = df[stock+' Weight'].max()
    max_tiker_portfolio = df.loc[df[stock+' Weight'] == max_ticker]
    ax1.scatter(x=max_tiker_portfolio['Volatility'],
                y=max_tiker_portfolio['Returns'], c='y', marker='*', s=100)
    ax1.annotate(stock, (max_tiker_portfolio['Volatility']*1.005,
                 max_tiker_portfolio['Returns']*0.995), size=12, color="white")

## パイチャート１ ###
df_pie1 = sharpe_portfolio.T.iloc[3:, :]
df_pie1 = df_pie1.sort_values(
    by=df_pie1.columns[0], axis=0, ascending=True, inplace=False)
col1 = [s.replace(' Weight', '') for s in df_pie1.index.tolist()]
ax2.pie(df_pie1.iloc[:, 0].tolist(),
        autopct="%1.1f%%", labels=col1, startangle=90)

## パイチャート２ ###
df_pie2 = min_variance_port.T.iloc[3:, :]
df_pie2 = df_pie2.sort_values(
    by=df_pie2.columns[0], axis=0, ascending=True, inplace=False)
col2 = [s.replace(' Weight', '') for s in df_pie2.index.tolist()]
ax3.pie(df_pie2.iloc[:, 0].tolist(),
        autopct="%1.1f%%", labels=col2, startangle=90)

## 積み上げグラフの元データ ###
df_all = (1+data.pct_change()).cumprod()

## 積み上げグラフ１ ###
df_sharpe = df_all.loc[:, col1]
for i in range(len(col1)):
    df_sharpe.iloc[:, i] = df_sharpe.iloc[:, i] * df_pie1.iloc[:, 0].values[i]
ax4.stackplot(df_sharpe.index, df_sharpe.values.T)
ax4.tick_params(axis='x', labelrotation=45)

## 積み上げグラフ２ ###
df_min = df_all.loc[:, col2]
for i in range(len(col2)):
    df_min.iloc[:, i] = df_min.iloc[:, i] * df_pie2.iloc[:, 0].values[i]
ax5.stackplot(df_min.index, df_min.values.T)
ax5.tick_params(axis='x', labelrotation=45)

plt.show()

## ポートフォリオをテキスト出力(ソートして出力) ###
print("\n---sharpe_portfolio-----\n")
sharpe_sort = sharpe_portfolio.iloc[:, 3:].T
sharpe_sort.sort_values(
    by=sharpe_sort.columns[0], ascending=False, axis=0, inplace=True)
print(sharpe_portfolio.iloc[:, 0:2].T)
print(sharpe_sort)

print("\n---min_variance_port-----\n")
min_sort = min_variance_port.iloc[:, 3:].T
min_sort.sort_values(
    by=min_sort.columns[0], ascending=False, axis=0, inplace=True)
print(min_variance_port.iloc[:, 0:2].T)
print(min_sort)

## ここまで ###
