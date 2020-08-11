import numpy as np

np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
filter = A_prev[0:2,0:2,:2]
# 最后一个2代表刚好取下标为2的数据
filter1 = A_prev[0:2,0:2,:,2]
print(filter)