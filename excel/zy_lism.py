import pandas as pd
import re


zy_excel = pd.read_excel('2020.08.21 Lism能力清单报价整理 -张煜.xlsx', sheet_name="农残")
sh_single = pd.read_excel('上海【农残部分】能力清单20200604.xlsx', sheet_name="上海单项农残清单")

# print(zy_excel)
i = 0
for index, row in zy_excel.iterrows():
    if i == 0:
        i=i+1
        continue
    print(index)
    if row["公开价格"] ==-1:
        test_name = row["测试名称"]
        test_method = row["检测方法名称"]
        p = r'.+-\d{4}'
        # gd = re.search(p, test_method).groupdict()
        fa = re.findall(p, test_method)
        strand = fa[0]
        res = sh_single.loc[sh_single['测试方法'].str.contains(strand, na=False)]
        res1 = sh_single[sh_single.化合物 == test_name]
        if not res.empty and  not res1.empty:
            print(test_name + strand+ "  in")
            zy_excel.at[index, row] = "380 + 100 * (N - 1)"
        else:
            print(test_name + strand+ " not in")



    i = i + 1



zy_excel.to_excel("device table.xlsx")
