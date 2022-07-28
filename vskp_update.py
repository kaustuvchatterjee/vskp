# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
from bs4 import BeautifulSoup
# import numpy as np
import re
import pandas as pd
from datetime import datetime, timedelta, date

############
#Procedures#
############

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#####################s

# Read imd webpage for vskp
url = "https://city.imd.gov.in/citywx/city_weather.php?id=43150"
r = requests.get(url, verify=False)
soup = BeautifulSoup(r.text,"html.parser")

# Parse data
data_table = soup.find('table', width='100%')
data = data_table.find_all('tr')
cols=[]
params=[]
values=[]
for i in range(len(data)-1):
    if data[i].find_all('td'):
        cols.append(data[i+1].find_all('td')[0].text.strip())
        values.append(data[i+1].find_all('td')[1].text.strip())

# Read csv file into data frame
df = pd.read_csv('vskp_data.csv')

# Update maximum temperature
try:
    i=0
    param = 'MaxTemp'
    str = cols[i]
    n=len(str)
    paramdate = datetime.strptime(str[n-9:n-1],'%d/%m/%y')
    value = values[i]
    if is_number(value):
        df = df.append({'obsdate':paramdate, 'param':param, 'value':value}, ignore_index=True)
except:
    pass

# Update minimum temperature
try:
    i=2
    param = 'MinTemp'
    str = cols[i]
    n=len(str)
    paramdate = datetime.strptime(str[n-9:n-1],'%d/%m/%y')
    value = values[i]
    if is_number(value):
        df = df.append({'obsdate':paramdate, 'param':param, 'value':value}, ignore_index=True)
except:
    pass


# Update precipitation
try:
    i=4
    param = 'precip'
    str = cols[i]
    n=len(str)
    paramdate = date.today()-timedelta(days=1)
    value = values[i]
    if is_number(value):
        df = df.append({'obsdate':paramdate, 'param':param, 'value':value}, ignore_index=True)
except:
    pass

# Update morning relative humidity
try:
    i=5
    param = 'RH'
    str = cols[i]
    n=len(str)
    pdate = date.today()
    pdate = pdate.strftime('%d/%m/%y')
    x = re.search('Humidity at\s',str)
    ptime = str[x.end():x.end()+4]
    pdate = pdate+":"+ptime
    paramdate = datetime.strptime(pdate,'%d/%m/%y:%H%M')
    value = values[i]
    if is_number(value):
        df = df.append({'obsdate':paramdate, 'param':param, 'value':value}, ignore_index=True)
except:
    pass

# Update evening relative humidity
try:
    i=6
    param = 'RH'
    str = cols[i]
    n=len(str)
    pdate = date.today()
    pdate = pdate.strftime('%d/%m/%y')
    x = re.search('Humidity at\s',str)
    ptime = str[x.end():x.end()+4]
    pdate = pdate+":"+ptime
    paramdate = datetime.strptime(pdate,'%d/%m/%y:%H%M')
    value = values[i]
    if is_number(value):
        df = df.append({'obsdate':paramdate, 'param':param, 'value':value}, ignore_index=True)
except:
    pass

# Write csv file
df.to_csv('vskp_data.csv',index=False)
