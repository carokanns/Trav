# %%
# !pip install -U streamlit
import streamlit as st
import time
import numpy as np
import datetime
import pandas as pd 
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from IPython.display import clear_output 
import altair as alt

plt.style.use('fivethirtyeight')

#%%
# SEKUSD = web.DataReader('DEXSDUS', 'fred')

# USDSEK = 1/SEKUSD.iloc[-1:,:1].values[0][0]
# st.write('USD-> SEK = ', round(USDSEK,3))
#%%
def getSEK():
  global EURSEK,EURUSD
  try:
      # print(f"try EUR->SEK") 
      EURSEK=web.DataReader('EURSEK=X','yahoo').iloc[-1]['Adj Close']
  except:  
      print(f"except EUR->SEK") 
      
  try:
      # print(f"try EUR->USD") 
      EURUSD = web.DataReader('EURUSD=X','yahoo').iloc[-1]['Adj Close']
  except:
      print(f"except EUR->USD") 
      
  USDSEK = EURSEK/EURUSD
  # print('SEK',USDSEK)
  return USDSEK
#%%
Innehav_btc = 0.0058832
Innehav_eth = 0.16534017
Innehav_bch = 0.52740029
Innehav_zrx = 166.38377079
Innehav_xrp = 259.351279

# startvärden om vi inte får kontakt
EURSEK=10.185529708862305 
EURUSD=1.1896264553070068 
SEK=getSEK()

BTC=60245.473
BCH=676.17
ETH=2145.07
ZRX=2.1237
XRP=1.3999


#%%
def getCrypto(init=False):
      # print(f"init={init}") 
  global BTC,BCH,ETH,ZRX,XRP
  if init:
    try: 
      BTCi = web.DataReader('BTC-USD','yahoo')['Adj Close'].tail(30) # Bitcoin
    except:
      BTCi = web.DataReader('BTC-USD','yahoo')['Adj Close'].tail(30) # Bitcoin
    # print('BTC klar')
    BTC = BTCi.iloc[-1]

    try: 
      BCHi = web.DataReader('BCH-USD','yahoo')['Adj Close'].tail(30) # Bitcoin cash
    except:
      BCHi = web.DataReader('BCH-USD','yahoo')['Adj Close'].tail(30) # Bitcoin cash
    # print('BCH klar')
    BCH=BCHi.iloc[-1]

    try: 
      ETHi = web.DataReader('ETH-USD','yahoo')['Adj Close'].tail(30) # Etherium
    except:
      ETHi = web.DataReader('ETH-USD','yahoo')['Adj Close'].tail(30) # Etherium
    # print('ETH klar')
    ETH=ETHi.iloc[-1]

    try: 
      ZRXi = web.DataReader('ZRX-USD','yahoo')['Adj Close'].tail(30) # 0x
    except:
      ZRXi = web.DataReader('ZRX-USD','yahoo')['Adj Close'].tail(30) # 0x
    # print('ZRX klar')
    ZRX=ZRXi.iloc[-1]

    try: 
      XRPi = web.DataReader('XRP-USD','yahoo')['Adj Close'].tail(30) # XRP
    except:
      XRPi = web.DataReader('XRP-USD','yahoo')['Adj Close'].tail(30) # XRP
    # print('XRP klar')
    XRP=XRPi.iloc[-1]

    crypto = {
    'BTC': BTCi, # Bitcoin
    'BCH': BCHi, # Bitcoin cash
    'ETH': ETHi, # Etherium
    'ZRX': ZRXi, # 0x
    'XRP': XRPi # XRP
    }
  else:      
    try: 
      BTC = web.DataReader('BTC-USD','yahoo').iloc[-1,:]['Adj Close'] # Bitcoin
      # print('BTC klar')
    except:
      pass
    
    try: 
      BCH = web.DataReader('BCH-USD','yahoo').iloc[-1,:]['Adj Close'] # Bitcoin cash
      # print('BCH klar')
    except:
      pass
    
    try: 
      ETH = web.DataReader('ETH-USD','yahoo').iloc[-1,:]['Adj Close'] # Etherium
      # print('ETH klar')
    except:
        pass

    try: 
      ZRX = web.DataReader('ZRX-USD','yahoo').iloc[-1,:]['Adj Close'] # 0x
      # print('ZRX klar')
    except:
      pass
      
    try: 
      XRP = web.DataReader('XRP-USD','yahoo').iloc[-1,:]['Adj Close'] # XRP
      # print('XRP klar')
    except:
      pass

    crypto = {
    'BTC': BTC, # Bitcoin
    'BCH': BCH, # Bitcoin cash
    'ETH': ETH, # Etherium
    'ZRX': ZRX, # 0x
    'XRP': XRP # XRP
    }

  return crypto #BTC,ETH,BCH,ZRX,XRP

def current():
  global df,SEK
  SEK = getSEK()
  minute=len(df)
  
  crypto = getCrypto()
  BTCac=crypto['BTC'] * Innehav_btc*SEK
  ETHac=crypto['ETH'] * Innehav_eth*SEK
  BCHac=crypto['BCH'] * Innehav_bch*SEK
  ZRXac=crypto['ZRX'] * Innehav_zrx*SEK
  XRPac=crypto['XRP'] * Innehav_xrp*SEK
  df.loc[len(df)] ={'Minute':minute, 'BTC':BTCac,'ETH':ETHac,'BCH':BCHac, 'ZRX':ZRXac, 'XRP':XRPac}  # lägg in som sista rad
  # print(df)
  return df


#%%
### init crypto 30 dagar ###
start=datetime.datetime.today().date()
print(start,'USD -> SEK',round(SEK,2))
print()
crypto = getCrypto(init=True)
df=pd.DataFrame(columns=['Minute'])
# df['BTC'] = crypto['BTC']* Innehav_btc * SEK
df['ETH'] = crypto['ETH']* Innehav_eth * SEK
# df['BCH'] = crypto['BCH']* Innehav_bch * SEK
# df['ZRX'] = crypto['ZRX']* Innehav_zrx * SEK
# df['XRP'] = crypto['XRP']* Innehav_xrp * SEK
df.reset_index(inplace=True)
df.drop('Date',axis=1,inplace=True)
df['Minute']=df.index.to_list()

#%%
fig, ax = plt.subplots()
x = df.Minute

line, = ax.plot(x, df.tail(30)['ETH'])
ax.set_ylim(df.ETH.min()*0.9,df.ETH.max()*1.1)

the_plot = st.pyplot(plt)
def init():  # give a clean slate to start
    line.set_ydata([np.nan] * len(x))

def animate(i):  # update the y values (every 1000ms)
    ax.set_title('ETH')
    line.set_ydata(df.tail(30).ETH)
    the_plot.pyplot(plt)
    

init()
print('efter init')
for i in range(100):
    df=current()
    print('efter current')
    x = df.tail(30).Minute
    
    animate(i)
    print('efter animate')
    time.sleep(30)

# df = df.tail(1)
# fig, ax = plt.subplots()
# ax.plot(df.Minute,df['ETH'])

# for i in range(420):
#   st.pyplot(fig)
#   df=current()
#   print(df.tail())
#   time.sleep(30)

# x = np.arange(100)
# source =df

# alt.Chart(df).mark_line().encode(
#     x='Minute',
#     y='ETH'
# )

# chart = st.line_chart(df['ETH'].tail(1))
# chart = st.altair_chart(df['ETH'],)
#%%
# for i in range(1, 101):
#     df=current()
#     new_rows = df['ETH'].tail(1)
#     chart.add_rows(new_rows)
#     last_rows = new_rows
#     time.sleep(30)

#%%
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
# st.button("Re-run")# %%


# %%
# model = CatBoostClassifier()
# model.load_model('modeller_bins_100'+'/'+'model_2021-02-13', format='cbm')
# print(model.feature_names_)

    
# %%
