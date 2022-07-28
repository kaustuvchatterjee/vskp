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
df['date'] = pd.to_datetime(df['date'])

# Update data frame
# date
date = date.today()

# maxTemp
if is_number(values[0]):
    maxTemp = values[0]

# minTemp
if is_number(values[2]):
    minTemp = values[2]

# mornRH
if is_number(values[5]):
    mornRH = values[5]

# eveRH
if is_number(values[6]):
    eveRH = values[6]
    
# meanRH
meanRH = (int(mornRH)+int(eveRH))/2

# precip
if is_number(values[4]):
    precip = values[4]
elif values[4] == 'NIL':
    precip = 0
print(precip)    
df = df.append({'date':date, 'maxTemp':maxTemp, 'minTemp':minTemp, 'mornRH':mornRH, 'eveRH':eveRH, 'meanRH':meanRH}, ignore_index=True)
if df[df['date']== date-timedelta(days=1)]['date'].count()==1:
    df.loc[df['date']== date-timedelta(days=1), 'precip'] =  precip

# Write csv file
df.to_csv('vskp_data.csv',index=False)
