#import numpy as np

#np.random.seed(シード値)
#y = np.random.randint(整数値の下限, 整数値の上限, 整数の個数)
#print(y)

#import matplotlib.pyplot as plt

#plt.plot(x軸の配列, y軸の配列)
#plt.show()

import numpy as np

np.random.seed(100)    #シード値は同じ乱数を表示してくれる。
x=np.arange(10)
y = np.random.randint(1,100, 10)
print(x)
print(y)


import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()
