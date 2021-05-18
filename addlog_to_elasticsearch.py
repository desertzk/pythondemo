import requests
import json



request_url = "http://10.103.13.151:9200/log_snwl_gui_1603334400/_doc/"


# headers={"content-type":"application/json"}
for i in range(50):

    payload = {
        "event": "Log",
        "source": "SonicWave",
        "model": "SONICWAVE 224W",
        "timestamp": 1618359734,
        "type": "Cloud Management",
        "subtype": "Cloud Command",
        "priority": "Notice",
        "tenant_id": "5de5d572d5f4aa0f3378c013",
        "tenant_name": "kazhang@sonicwall.com@sonicwall.com",
        "device": "2c:b8:ed:09:f3:e7",
        "data": {
            "description": "------duplicate  duplicate  duplicate----- "+str(i)
        }
    }
    payload["timestamp"] = 1618359734
    result = requests.post(request_url, json=payload)
    print(result.status_code)
    print(result.text)