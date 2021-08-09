import numpy as np
import pandas as pd
from datetime import datetime as dt
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import streamlit as st
from IPython.display import clear_output 
# from IPython.display import clear_output 
# from catboost import CatBoostRegressor,Pool,utils

import time
plt.style.use('fivethirtyeight')
def plot_ETH(ETH):
    ### plot Adj Close ###
    x=ETH.tail(90).index
    y=ETH.tail(90)['Adj Close']     
    plt.figure(figsize=(12,10))
    fig, ax = plt.subplots()
    plt.xticks(rotation=45)

    line, = ax.plot(x, y)

    the_plot = st.pyplot(plt)
    ax.set_title('ETH')
        
    for r in range(5):
        plt.pause(10)
        ETH = web.DataReader('ETH-USD','yahoo') # Etherium
        line.set_xdata(ETH.tail(90).index)
        line.set_ydata(ETH.tail(90)['Adj Close'])  
        the_plot.pyplot(plt)

    

load = st.beta_container()
graf = st.beta_container()


with load:
    if st.button('load'):
        ETH = web.DataReader('ETH-USD','yahoo') # Etherium
        clear_output()
        st.write("klar")
        st.title('ETH')
        plot_ETH(ETH.copy())
        # plt.show()
        
    else:
        st.write('## Ladda data')    

    
    
