from multiprocessing import Pool
import time
import random
import os

a = 1

def f(x):
    interval = random.randrange(1,5)
    print(f"{os.getpid()} sleeping for {interval}")
    time.sleep(interval)
    print(f"{os.getpid()} just woke up")
    return x*x

with Pool(4) as pool:
    pow10 = pool.map(f, range(10,20))

print(pow10)