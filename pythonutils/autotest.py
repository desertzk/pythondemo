import json
import requests
import copy
import random


class JsonDecoder:
    def __init__(self):
        self.value = None
        self.list = []
        self.dict = {}

    def __getvalue__(self, dictionary, key):
        for k in dictionary:
            #print k
            if k == key and dictionary[k]!=[]:
                self.value = dictionary[k]
                return
            if isinstance(dictionary[k], dict):
                self.__getvalue__(dictionary[k], key)
            if isinstance(dictionary[k], list):
                for i in range(len(dictionary[k])):
                    if isinstance(dictionary[k][i], dict):
                        self.__getvalue__(dictionary[k][i], key)
        # return self.value

    def getValueFromDictByKey(self, dictionary, key):
        if isinstance(dictionary, list):
            dictionary = {"results": dictionary}
        self.__getvalue__(dictionary, key)
        return self.value


    def getValueFromDictBy2Key(self, dictionary, key1,key2):
        if isinstance(dictionary, list):
            dictionary = {"results": dictionary}
        self.__getvalue__(dictionary, key1)
        self.getValueFromDictByKey(self.value, key2)
        return self.value






class JsonEncoder:
    def __init__(self):
        self.value = None
        self.list = []
        self.dict = {}

    def __setvalue__(self, dictionary, key, value):
        for k in dictionary:
            # print k
            if k == key:
                dictionary[k] = value
            if isinstance(dictionary[k], dict):
                self.__setvalue__(dictionary[k], key, value)
            if isinstance(dictionary[k], list):
                for i in range(len(dictionary[k])):
                    if isinstance(dictionary[k][i], dict):
                        self.__setvalue__(dictionary[k][i], key, value)

    def __removeKey__(self, dictionary, modifyKeyName, key):
        for k in dictionary:
            if k == key:
                newdict = copy.deepcopy(dictionary)
                newdict.pop(key)
                self.list.append(newdict)
                print(self.list)
            if isinstance(dictionary[k], dict):
                self.__removeKey__(dictionary[k], modifyKeyName, key)
            if isinstance(dictionary[k], list):
                for i in range(len(dictionary[k])):
                    if isinstance(dictionary[k][i], dict):
                        self.__removeKey__(dictionary[k][i], modifyKeyName, key)

    def replaceListValueByKey(self, dictionary, newKeyName, replacekeyName, newValue):
        if isinstance(dictionary, list):
            dictionary = {newKeyName: dictionary}
        self.__setvalue__(dictionary, replacekeyName, newValue)
        return dictionary

    def removeValueAndKey(self, dictionary, modifyKeyName, keyName):
        self.__removeKey__(dictionary, modifyKeyName, keyName)
        self.__setvalue__(dictionary, modifyKeyName, self.list)
        print(dictionary)
        return dictionary

    def transferDictToJson(self, dictionary):
        str_json = json.dumps(dictionary)
        print(type(str_json))
        print(str_json)
        return str_json

    def randomStringStartwitchPerfix(self, perfix):
        return perfix+str(random.randint(0, 99))







# response=requests.get("http://10.103.13.225:80/api/v2/tenants/5de5d572d5f4aa0f3378c013/network/locations",headers={"X-Amz-Security-Token": "5de5d572d5f4aa0f3378c012"})
#
# print(response)

jsd = JsonDecoder()

jsondict = {
    "secret": "cNrk3Ytq2ywWfdSID3zs2CXwVFrK6t4z",
    "chicken": {
        "version": 61177166
    },
    "panda": {
        "cpuUsage": False,
        "memUsage": False,
        "throughput24": False,
        "throughput5": False,
        "clientNum": False,
        "datapack": True
    },
    "apiGateway": {
        "raccoonUrl": "http://10.103.13.225",
        "chickenUrl": "http://10.103.13.225",
        "pandaUrl": "10.103.13.225",
        "tortoiseUrl": "https://tortoise-sonicwall.sonicwall.com"
    },
    "updateFirmware": {
        "version": "Not Found",
        "is_auto": True,
        "retry_duration": 3600,
        "schedule": [
            "00 07 * * 0"
        ]
    }
}
version = jsd.getValueFromDictBy2Key(jsondict,"chicken","version")
print(version)

response=requests.get("http://10.103.13.225:80/api/v2/tenants/5de5d572d5f4aa0f3378c013/policies/ssid-groups",headers={"X-Amz-Security-Token": "5de5d572d5f4aa0f3378c012"})
jed = JsonEncoder()

resp = json.loads(response.text)
ssids = jsd.getValueFromDictByKey(resp,"ssids")
groupID = jsd.getValueFromDictByKey(resp,"default_group_id")
print(groupID)
ssidName = jed.randomStringStartwitchPerfix("sonicwall")

ssids = jed.replaceListValueByKey(ssids,"ssid_profiles","ssid_name",ssidName)
print(ssids)
ssids=jed.removeValueAndKey(ssids,"ssid_types","ssid_id")
print(ssids)
json_body=jed.transferDictToJson(ssids)
print(json_body)