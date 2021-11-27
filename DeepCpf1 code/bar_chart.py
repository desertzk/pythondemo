import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['G1']
RandomForest_aucs = [0.749]
GradientBoost_aucs = [0.830]
Seq_deepcpf1_aucs = [0.870]
Crispr_ont_aucs = [0.808]
ReinforceNasCRISPR_aucs = [35]
# WT_means = [25]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 2.5*width, RandomForest_aucs, width, label='SpCas9')
rects2 = ax.bar(x - 1.5*width, xCas9_aucs, width, label='xCas9')
rects3 = ax.bar(x - 0.5*width, SniperCas_aucs, width, label='SniperCas')
rects4 = ax.bar(x + 0.5*width, eSpCas_aucs, width, label='eSpCas')
rects5 = ax.bar(x + 1.5*width, HF1_aucs, width, label='HF1')
# rects6 = ax.bar(x + 2.5*width, WT_means, width, label='WT')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
# autolabel(rects6)

fig.tight_layout()

plt.show()