
import matplotlib.pyplot as plt

truelist =["A","B","C","D","E","F","G","H","I","J"]

L1=["A","L","B","N","O","P","Q","C","S","T","D","V","W","X","Y"]
L2=["K","A","M","N","O","B","Q","R","S","T","C","V","W","X","D"]

plist = []
rlist = []
i = 0
for item in L1:
    i = i + 1
    tempL1 = L1[0:i]
    sameitem = set(tempL1) & set(truelist)
    num = len(sameitem)
    tp = num
    # precision = tp /(tp+fp) precision是相对你自己的模型预测而言：true positive ／retrieved set。假设你的模型一共预测了100个正例，而其中80个是对的正例，那么你的precision就是80%。我们可以把precision也理解为，当你的模型作出一个新的预测时，它的confidence score 是多少，或者它做的这个预测是对的的可能性是多少。
    precision = tp / i
    # recall =tp/(tp+fn) recall 是相对真实的答案而言： true positive ／ golden set 。假设测试集里面有100个正例，你的模型能预测覆盖到多少，如果你的模型预测到了40个正例，那你的recall就是40%
    recall = tp / len(truelist)

    plist.append(precision)
    rlist.append(recall)


i=0
plist2=[]
rlist2=[]
for item in L2:
    i = i + 1
    tempL2 = L2[0:i]
    sameitem = set(tempL2) & set(truelist)
    num = len(sameitem)
    tp = num
    # precision = tp /(tp+fp)
    precision = tp / i
    # recall =tp/(tp+fn)
    recall = tp / len(truelist)

    plist2.append(precision)
    rlist2.append(recall)


print(plist2)
print(rlist2)
print(recall)
print(plist)


# dates = ['2015-09-12','2015-09-22','2015-12-10','2015-12-20','2015-12-22']
# PM_25 = [80, 55,100,45,56]
# dates = [pd.to_datetime(d) for d in dates]
# plt.xlabel(rlist)
# plt.ylabel(plist)
plt.plot(rlist2, plist2)
plt.show()


