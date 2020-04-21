import json
import requests



class JsonDecoder:
    def __init__(self):
        self.value = None
        self.list = []
        self.dict = {}

    def __getvalue__(self, dictionary, key):
        for k in dictionary:
            # print k
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


    def getValueFromDictBy2Key(self, dictionary, key,key2):
        if isinstance(dictionary, list):
            dictionary = {"results": dictionary}
        self.__getvalue__(dictionary, key)
        self.getValueFromDictByKey(self.value, key2)
        return self.value



response=requests.get("http://10.103.13.225:80/api/v2/tenants/5de5d572d5f4aa0f3378c013/network/locations",headers={"X-Amz-Security-Token": "5de5d572d5f4aa0f3378c012"})

print(response)
jsd = JsonDecoder()
jsd.getValueFromDictBy2Key(json.loads(response.text),"zones","id")