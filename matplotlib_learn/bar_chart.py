import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyquery import PyQuery
f = open("CRISPR RGEN Tools.html","r", encoding="utf-8")
html = f.read() # Your HTML CODE
pq = PyQuery(html)
tag = pq('div#subtable table') # or     tag = pq('div.class')
crRNA = pq('span.input-rgenseq').text()
listcrrna = crRNA.join(',')
# print(tag.text())
print(str(crRNA))
dfs = pd.read_html(str(tag))
df = dfs[0]
base = df.iloc[2]
base.dropna()
base.intersection(listcrrna)
countries = ['G', 'A', 'A', 'C', 'A']
bronzes = np.array([138, 17, 26, 19, 15])
silvers = np.array([37, 123, 18, 78, 10])
golds = np.array([46, 27, 26, 119, 17])
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