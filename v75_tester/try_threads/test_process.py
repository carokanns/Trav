#%%
import concurrent.futures
from multiprocessing.dummy import freeze_support
import pandas as pd
import numpy as np
import math
import time
#%%
# load all_data.csv
all_data = pd.read_csv('../all_data.csv')
#%%

def my_func(x):
    for i in range(x):
        for index, row in all_data.iterrows():
            x = np.sqrt(row.avd) + np.sqrt(row.spår) + np.sqrt(row.vodds) + \
                np.sqrt(row.podds)+np.sqrt(row.kr)+np.tanh(np.sqrt(row.streck))
    # print(x)

    return x
#%%

def doit():
    # make threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        y = executor.map(my_func, [2, 3])
    return y

def meassure_doit():
    start_time = time.time()
    y = doit()
    print("doit:  --- %s seconds ---" % (time.time() - start_time))
    print('tuple in meassure', tuple(y))

#%%
PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def doit_prime():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

def main():
    print('start prime')
    start_time = time.time()
    doit_prime()    
    print("prime:  --- %s seconds ---" % (time.time() - start_time))

    print('start doit')
    meassure_doit()
    print('end')
            
            

if __name__ == '__main__':
    # freeze_support()
    main()
    
