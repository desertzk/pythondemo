#!/usr/bin/python3

import concurrent.futures
import subprocess
import sys

def try_password(aPassword, aFile, aExecutor):
  myRet = subprocess.call(["7z", "t", "-p\"" + aPassword + "\"", aFile], stdout=subprocess.DEVNULL)
  if myRet == 0:
    print ("The password is: " + aPassword)



myFilename = sys.argv[2]
myPasswordFile = sys.argv[1]


#iterate through each character permutation, unless stopThis variable is True
def permutate(length, position, base):
    for character in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']:
        # if main.stopThis == True:
        #     return
        if (position < length - 1):
            permutate(length, position + 1, base + character)
        if len(base) + 1 == length:
            myExecutor.submit(try_password, base + character, myFilename, myExecutor)
            # check(base + character)





print ("File to crack: " + myFilename)
print ("Password list: " + myPasswordFile)

myFile = open(myPasswordFile, "r")
myExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

for baseWidth in range(1, 9):
  permutate(baseWidth, 0, '')

i=0
for myLine in iter(myFile):
  i=i+1
  print(str(i))
  print(str(myLine))
  myPassword = myLine.strip('\n')
  myExecutor.submit(try_password, myPassword, myFilename, myExecutor)
