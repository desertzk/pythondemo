import numpy as np
import pandas as pd
import json


device_data = pd.read_csv('manage_device.txt', sep="    ")
device_data=device_data.set_index('serial_number')
device_data['online status'] = ''
with open('resp.txt') as json_file:
    data = json.load(json_file)

    for device in data:
        connection_event = device.get("value",{}).get("connection_event",{})

        status = connection_event.get("value",{}).get("status","offline")
        try:
            device_mac = device.get("full").split("/")[1]
            mac_list=device_mac.split(":")
            separator = ""
            serial_number = separator.join(mac_list)
            device_data.at[serial_number,"online status"] =status
        except Exception as ex:
            print("-------------------------------------------------"+device.get("full"))



        print(device_mac+"   "+status)

print(device_data)
device_data.to_excel("device table.xlsx")
