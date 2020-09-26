import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyquery import PyQuery


#A to T 颜色
A_T_color='#CCFF33'
#A to G 颜色
A_G_color='#9933CC'
#A to C 颜色
A_C_color='#3399FF'


f = open("CRISPR RGEN Tools.html","r", encoding="utf-8")
html = f.read() # Your HTML CODE
pq = PyQuery(html)
tag = pq('div#subtable table') # or     tag = pq('div.class')
crRNA = pq('span.input-rgenseq').text()
crRNA = crRNA.split(' ')[0]
listcrrna = [char for char in crRNA if char != ' ']
# print(tag.text())
print(str(crRNA))
dfs = pd.read_html(str(tag))
df = dfs[0]
base = df.iloc[2:]
base.columns = base.iloc[0]
base = base.iloc[1:]
col_index = base.iloc[:,0]
base = base.set_index(col_index)
base = base.iloc[:,1:]
# bases = base.dropna()
# need_base = base.loc[:,listcrrna]


base_list = list(base.columns)
base_str = "".join(base_list)

position = base_str.index(crRNA)
base = base.iloc[:,position:position+len(listcrrna)]

col = 0
A_list = []
T_list = []
G_list = []
C_list = []
index = 0
base_pos_dict ={"A":0}

# for critem in listcrrna:
#     if critem != 'A':
#         continue
#     A_T =

changes_from_adenine_dict = {}
while index < len(listcrrna):
    item = listcrrna[index]
    if item != 'A':
        changes_from_adenine_dict[index] = [0,0,0]
    else:
        A_T = float(base.iloc[1,index])
        A_G = float(base.iloc[2,index])
        A_C = float(base.iloc[3,index])
        sum = A_T + A_G + A_C
        if sum == 0:
            changes_from_adenine_dict[index] = [0, 0, 0]
        else:
            changes_from_adenine_dict[index] = [A_T/sum,A_G/sum,A_C/sum]


        # while col < base.columns.size:
        #     # col_item = base.columns[col_index]
        #     basecol = base.columns[col]
        #     if item == basecol:
        #
        #         A_list.append(float(base.iloc[0,col]))
        #         T_list.append(float(base.iloc[1,col]))
        #         G_list.append(float(base.iloc[2,col]))
        #         C_list.append(float(base.iloc[3,col]))
        #         index = index + 1
        #         col = col + 1
        #         break

    index = index + 1




percent_df = pd.DataFrame.from_dict(changes_from_adenine_dict, orient='index',columns=["A_T","A_G","A_C"])
# retbase = base.intersection(listcrrna)
countries = listcrrna
A2T = percent_df["A_T"]
A2G = percent_df["A_G"]
A2C = percent_df["A_C"]
ind = [x for x, _ in enumerate(countries)]

# plt.bar(ind, bronzes, width=0.8, label='A to A', color='blue')
plt.bar(ind, A2C, width=0.8, label='A to T', color=A_T_color, bottom=A2G+A2T)
plt.bar(ind, A2G, width=0.8, label='A to G', color=A_G_color, bottom=A2T)
plt.bar(ind, A2T, width=0.8, label='A to C', color=A_C_color)


plt.xticks(ind, countries)
plt.ylabel("the ratio of nucleotide changes from adenine at each position(%)")
plt.xlabel("DNA sequences targeted by guide RNA")
plt.legend(loc="upper right")
plt.title("A to B substitution in target windows")


plt.savefig('./adenine.svg', format='svg')
plt.show()
