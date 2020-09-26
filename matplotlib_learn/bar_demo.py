import matplotlib.pyplot as plt
import numpy as np

countries = ['G', 'A', 'A', 'C', 'A']
bronzes = np.array([0, 17, 26, 19, 15])
silvers = np.array([0, 123, 18, 78, 10])
golds = np.array([0, 27, 26, 119, 17])
ind = [x for x, _ in enumerate(countries)]

plt.bar(ind, golds, width=0.8, label='A to T', color='gold', bottom=silvers+bronzes)
plt.bar(ind, silvers, width=0.8, label='A to G', color='silver', bottom=bronzes)
plt.bar(ind, bronzes, width=0.8, label='A to C', color='#CD853F')

plt.xticks(ind, countries)
plt.ylabel("the ratio of nucleotide")
plt.xlabel("DNA sequences targeted by guide RNA")
plt.legend(loc="upper right")
plt.title("A to B substitution in target windows")

plt.show()