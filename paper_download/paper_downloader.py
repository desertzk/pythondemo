# coding:utf-8


import re
import requests
import urllib
import os

# get web context
r = requests.get('http://openaccess.thecvf.com/ICCV2019.py')
data = r.text
# find all pdf links
link_list = re.findall(r"(?<=href=\").+?pdf(?=\">pdf)|(?<=href=\').+?pdf(?=\">pdf)" ,data)
name_list = re.findall(r"(?<=href=\").+?2019_paper.html\">.+?</a>" ,data)
cnt = 0
num = len(link_list)
print('num=',num)
# your local path to download pdf files
localDir = 'D:\papers\ICCV2019\\'
if not os.path.exists(localDir):
    os.makedirs(localDir)



while cnt < num:
    url = link_list[cnt]
    # seperate file name from url links
    file_name = name_list[cnt].split('<')[0].split('>')[1]
    # to avoid some illegal punctuation in file name
    file_name = file_name.replace(':','_')
    file_name = file_name.replace('\"','_')
    file_name = file_name.replace('?','_')
    file_name = file_name.replace('/','_')
    file_path = localDir + file_name + '.pdf'
    # download pdf files
    print('['+str(cnt)+'/'+str(num)+"]  Downloading -> "+file_path)
#    try:
    urllib.request.urlretrieve('http://openaccess.thecvf.com/'+url,file_path)
#    except Exception:
#        continue
    cnt = cnt + 1
print("all download finished")

