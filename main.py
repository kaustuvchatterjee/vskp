#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Tue Apr 27 12:09:34 2021
"""
@author: kaustuv
"""
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2gray #, rgb2hsv
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.restoration import inpaint
from skimage import filters
import pandas as pd
import matplotlib.dates as mdates
from pvlib import solarposition
from datetime import datetime
import pytz
import pytesseract

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from PIL import Image, ImageEnhance
from PIL import ImageFont
from PIL import ImageDraw

import urllib.request
import requests, json, datetime



#---------------------------------
def read_data():
    df = pd.read_csv('vskp_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df['date'], df['maxTemp'], df['minTemp'], df['relHum'], df['rainFall']

t, maxTemp, minTemp, relHum, precip = read_data()

#-------------------------------------

# Calculate Heat Index
meanTemp = (minTemp + maxTemp)/2

T = (meanTemp*9/5)+32;
R = relHum;
c1 = -42.379;
c2 = 2.04901523;
c3 = 10.14333127;
c4 = -0.22475541;
c5 = -6.83783e-3;
c6 = -5.481717e-2;
c7 = 1.22874e-3;
c8 = 8.5282e-4;
c9 = -1.99e-6;
HI = c1+c2*T+c3*R+c4*T*R+c5*T**2+c6*R**2+c7*T**2*R+c8*T*R**2+c9*T**2.*R**2;

heatIndex = (HI-32)*5/9;

# Temperature Plot
tx = np.hstack((t,t[::-1]))
tempy = np.hstack((maxTemp,minTemp[::-1]))

fig1 = plt.figure(figsize=(12,6))
plt.fill_between(t,minTemp, maxTemp, color='lightblue', alpha = 0.6)
plt.plot(t,meanTemp, t,heatIndex)
yTitle = "Temperature ($^\circ$C)"
plt.ylabel(yTitle, fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.title ('Temperature')
plt.grid()
label=['Max/Min Temp','Mean Temp','Heat Index']
plt.legend(label)
plt.ylim([0,50])
ax=plt.gca()
monthyearFmt = mdates.DateFormatter('%d %b %y')
ax.xaxis.set_major_formatter(monthyearFmt)
plt.autoscale(enable=True, axis='x', tight=True)


st.pyplot(fig1, dpi=300)

# Precipitation Plot
# fig2 = make_subplots(specs=[[{"secondary_y": True}]])
# fig2.add_trace(go.Scatter(x=t,y=precip, mode="lines", name="Precipitation",line={'dash': 'solid', 'color': 'dodgerblue'}),
#                secondary_y=False)
# fig2.add_trace(go.Scatter(x=t,y=relHum, mode="lines", name="Rel Humidity",line={'dash': 'solid', 'color': 'lightseagreen'}),
#                secondary_y=True)

# fig2.update_layout(title_text = 'Precipitation & Relative Humidity',
#                 xaxis_title='Date',
#                 width = 740, height=480,
#                 margin=dict(r=20, b=10, l=10, t=30),
#                 showlegend = True,
#                 template = 'plotly_white'
#                 )
# fig2.update_yaxes(title_text="Precipitation (mm)", 
# #                  range = [0,100],
#                   secondary_y=False)
# fig2.update_yaxes(title_text="Relative Humidity (%)", 
#                   range = [0,100],
#                   secondary_y=True)

# fig2.update_layout(legend=dict(
#     yanchor="top",
#     y=0.99,
#     xanchor="left",
#     x=0.01,
#     bgcolor = 'rgba(255,255,255,0.8)'
# ))

fig2 = plt.figure(figsize=(12,6))
plt.grid()
plt.plot(t[:-1],relHum[:-1], color='teal')
plt.fill_between(t[:-1],relHum[:-1], color = 'lightgreen', alpha=0.2)
ax1 = plt.gca()
ax1.set_ylim([0,100])
ax1.yaxis.set_ticks_position('left')
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.spines['left'].set_color('teal')
ax1.spines['left'].set_position(('axes',-0.02))
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_position(('data',0))
ax1.set_ylabel('Relative Humidity (%)',
              fontsize=12)


plt.twinx()

plt.plot(t,precip, color='#1f77b4')
plt.fill_between(t,precip, color='lightblue', alpha = 0.8)
ax2 = plt.gca()
ax2.set_ylim([0,500])
ax2.yaxis.set_ticks_position('left')
ax2.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.spines['left'].set_position(('axes',-0.1))
ax2.spines['left'].set_linewidth(2)
ax2.spines['bottom'].set_position(('data',0))
ax2.set_ylabel('Precipitatiom (mm)',
              fontsize=12,)

ax2.yaxis.set_label_position("left")
plt.grid()

      
# monthyearFmt = mdates.DateFormatter('%b %y')
monthyearFmt = mdates.DateFormatter('%d %b %y')
ax2.xaxis.set_major_formatter(monthyearFmt)
plt.autoscale(enable=True, axis='x', tight=True)
plt.title('Relative Humidity & Precipitation')
plt.xlabel('Date', fontsize=12,)
# st.plotly_chart(fig2)
st.pyplot(fig2, dpi=300)

tz = 'Asia/Calcutta'
try:
    # Doppler Radar Plot
    url = 'https://mausam.imd.gov.in/Radar/sri_vsk.gif'
    img1 = io.imread(url)
    img1_shape = np.shape(img1)
    # Satellite image (water vapor)
    url = 'https://mausam.imd.gov.in/Satellite/3Dasiasec_wv.jpg'
    img2 = io.imread(url)


    # Extract Date/Time
    img1_dt = img1[25:45,520:688,:]
    img2_dt = img2[30:50,510:790,:]
    try:
        local_tz = pytz.timezone(tz)
        text = pytesseract.image_to_string(img1_dt)
        dp_datetime = datetime.strptime(text.strip(),'%H:%M / %d-%b-%Y')
        dp_datetime = dp_datetime.replace(tzinfo=pytz.utc).astimezone(local_tz)
        dp_datetime = datetime.strftime(dp_datetime,'%d %b %Y %H:%M IST')

    except:
        dp_datetime = "NA"

    try:
        text = pytesseract.image_to_string(img2_dt).strip()
        text = text[:10]+' '+text[20:24]
        sat_datetime = datetime.strptime(text.strip(),'%d-%m-%Y %H%M')
        sat_datetime = datetime.strftime(sat_datetime,'%d %b %Y %H:%M IST')

    except Exception as e:
        print(e)
        sat_datetime = "NA"


    # Crop


    img1 = img1[:, 0:500, 0:3]
    x1 = 730
    x2 = x1+41
    y1 = 762
    y2 = y1+41
    img2 = img2[x1:x2,y1:y2,:]
    img2 = resize(img2,(500,500))

    mask = np.load('mask_vsk.npy')
    element = np.ones([31,31],np.uint8)
    mask = dilation(mask,element)
    masked = np.where(mask[...,None], img2, 0)

    result = img2.copy()
    result[mask>0]=(0,0,0)
    img2_gray = rgb2gray(result)
    img2_gray = inpaint.inpaint_biharmonic(img2_gray,mask)
    img2_gray = filters.gaussian(img2_gray,sigma=1)
    alpha = img2_gray

    dpimg = img1
    dp_dt = img1_dt
    sat_dt = img2_dt
    #Plot
    bbox=dict(boxstyle="square", alpha=0.3, color='white')
    fig3, ax = plt.subplots(figsize=[15,15])
    ax.set(xticks=[], yticks=[], title="Visakhapatnam Doppler Radar Image Overlayed with Satellite Image")
    plt.imshow(dpimg)
    plt.imshow(img2_gray, cmap='Blues_r', alpha=alpha*0.8)

    plt.annotate('Radar:      '+dp_datetime,(364,14),size=11, color = 'k', fontweight='semibold', bbox=bbox)
    plt.annotate('Satellite:  '+sat_datetime,(364,24),size=11, color = 'k', fontweight='semibold', bbox=bbox) 

    # dpin = ax.inset_axes([360,2,70,12],transform=ax.transData)    # create new inset axes in data coordinates
    # dpin.imshow(dp_dt, cmap='gray', alpha = 0.8)
    # dpin.axis('off')

    # satin = ax.inset_axes([360,14,120,10],transform=ax.transData)    # create new inset axes in data coordinates
    # satin.imshow(sat_dt, cmap = 'gray', alpha = 0.8)
    # satin.axis('off')

    st.pyplot(fig3)    

except:
    st.text("Unable to load Radar & Satellite images!")

#-------------------------------------------------------
lat, lon = 17.6744, 83.284

times = pd.date_range('2019-01-01 00:00:00', '2020-01-01', closed='left',
                      freq='H', tz=tz)
solpos = solarposition.get_solarposition(times, lat, lon)
# remove nighttime
solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

fig, ax = plt.subplots(figsize=[12,12])
ax = plt.subplot(1, 1, 1, projection='polar')
# draw hour labels
for hour in np.unique(solpos.index.hour):
    # choose label position by the smallest radius for each hour
    subset = solpos.loc[solpos.index.hour == hour, :]
    r = subset.apparent_zenith
    pos = solpos.loc[r.idxmin(), :]
    # ax.text(np.radians(pos['azimuth']), pos['apparent_zenith'], str(hour))
# draw individual days
colors = ['g','r','b','y']
lws = [4,4,4,8]
curdate = pd.Timestamp.today()
curdate = str(curdate)[0:10]
i=0
for date in pd.to_datetime(['2022-03-21', '2022-06-21', '2022-12-21',curdate]):
    times = pd.date_range(date, date+pd.Timedelta('24h'), freq='5min', tz=tz)
    solpos = solarposition.get_solarposition(times, lat, lon)
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
    label = date.strftime('%Y-%m-%d')
    ax.plot(np.radians(solpos.azimuth), solpos.apparent_zenith, label=label, lw=lws[i], color=colors[i])
    i+=1

curtime = pd.Timestamp.now(tz=tz)
if curtime.strftime('%p')=='AM':
    delta = -0.2
else:
    delta = 0.2
cursolpos = solarposition.get_solarposition(curtime, lat, lon)
anText = 'Az: %s$^\degree$\nEl: %s$^\circ$' % (str(np.round(cursolpos.azimuth[0],1)), str(np.round(cursolpos.apparent_elevation[0],1)))
if cursolpos.apparent_elevation[0] > 0:
    ax.plot(np.radians(cursolpos.azimuth), cursolpos.apparent_zenith, marker='o', markersize=30, color='orange')
    ax.plot([np.radians(cursolpos.azimuth[0]),0],[cursolpos.apparent_zenith[0],0], lw=8, color='orange')
    ax.text(np.radians(cursolpos.azimuth[0])+delta,cursolpos.apparent_zenith[0]+0, anText, fontsize=24, fontweight='bold', color='#cfea15',horizontalalignment='center')
# ax.figure.legend(loc='upper left')

# change coordinates to be like a compass
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rmax(90)

ax.set_xticks([])
ax.set_yticks([])

# ax.plot([0,0],'x', color='k')
plt.savefig('sp.png', transparent=True)
# plt.show(ax)

bg = io.imread('dh.png')
fg = io.imread('sp.png')
fg = resize(fg,(1460,1460))

fig4, ax = plt.subplots(figsize=[12,12])
ax.set(xticks=[], yticks=[], title="Sun Path and Position on "+curtime.strftime('%d %b %Y')+" at "+curtime.strftime('%H:%M'))

plt.imshow(bg)
plt.imshow(fg)
st.pyplot(fig4) 

#---------------------------------
# Current Weather
def parseData(data):

    attrList = data.split(';')
    h = attrList[0]
    h = h.split(':')[1].strip()
    h = int(h.split('px')[0].strip())

    w = attrList[1]
    w = w.split(':')[1].strip()
    w = int(w.split('px')[0].strip())

    l = attrList[2]
    l = l.split(':')[1].strip()
    l = int(l.split('px')[0].strip())

    t = attrList[3]
    t = t.split(':')[1].strip()
    t = int(t.split('px')[0].strip())

    return [h,w,l,t]

def createImageFromTiles(s, l, t, h, w):
    # print(s,l,t)
    min_l = np.min(l)
    max_l = np.max(l)+w
    min_t = np.min(t)
    max_t = np.max(t)+h
    img = Image.new('RGBA',(max_l, max_t))
    # print(min_l, max_l, min_t, max_t)

    
    for i in range(len(s)):
        
        urllib.request.urlretrieve(s[i], 'tmp.png')
        fmg = Image.open('tmp.png')
        fmg.resize((h,w))
        # fmg.show()
        img.paste(fmg,(l[i]-min_l,t[i]-min_t))
        # img.show()
        
    return img

def getLayer(url, Xpath):

    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=1920,1080")
    timeout = 10
    browser = webdriver.Chrome(options=options)
    browser.get(url)
    element_present = EC.visibility_of_all_elements_located((By.XPATH, Xpath))
    ImageList = WebDriverWait(browser, timeout).until(element_present)

    s = []
    l = []
    t = []
    for element in ImageList:

        src = element.get_attribute('src')
        data = element.get_attribute('style')
        # print(data)
        img_data = parseData(data)
        l.append(img_data[2])
        t.append(img_data[3])
        s.append(src)
        h = img_data[0]
        w = img_data[1]
        
    img = createImageFromTiles(s,l,t,h,w)
    
    return img

# Base Map
url = 'https://openweathermap.org/weathermap?basemap=map&cities=false&layer=clouds&lat=17.69&lon=83.2093&zoom=8'
Xpath = '//*[@id="map"]/div[1]/div[1]/div[2]/div[2]/*'
base_img = getLayer(url, Xpath)

# # Labels
# Xpath = '//*[@id="map"]/div[1]/div[1]/div[3]/div[2]/*'
# labels_img = getLayer(url, Xpath)

# Clouds
Xpath = "//*[@id='map']/div[1]/div[1]/div[1]/div[2]/*"
clouds_img = getLayer(url, Xpath)

# Radar
url = 'https://openweathermap.org/weathermap?basemap=map&cities=false&layer=radar&lat=17.69&lon=83.2093&zoom=8'
radar_img = getLayer(url, Xpath)

img = base_img.copy()
img.paste(clouds_img, (0,0), clouds_img)
img.paste(radar_img, (0,0), radar_img)
img = img.convert('RGB')

left = 672
top = 374
right = 1272
bottom = 774

img = img.crop((left, top, right, bottom))
# img.save('wmap.jpg')

api_key = "85d5db5a1265e695a3d4b99399f27d57"
base_url = "http://api.openweathermap.org/data/2.5/weather?"
city_name = "Visakhapatnam,in"

complete_url = base_url + "appid=" + api_key + "&q=" + city_name
response = requests.get(complete_url)
x = response.json()

temp = 'NA'
RH = 'NA'
pressure = 'NA'
wind_speed = 'NA'
wind_dir = 'NA'
clouds = 'NA'
w_desc = 'NA'



if x["cod"] != "404":


    y = x["main"]
    temp = y["temp"]
    temp = np.round(temp-273.15,1)
    minTemp = np.round((y["temp_min"]-273.15),0)
    maxTemp = np.round((y["temp_max"]-273.15),0)

    mslp = y["pressure"]
    RH = y["humidity"]

    z = x["weather"]
    w_desc = z[0]["description"]
    w_icon = "https://openweathermap.org/img/wn/"+z[0]["icon"]+"@2x.png"
    urllib.request.urlretrieve(w_icon, 'wimg.png')
    wicon = Image.open('wimg.png')

    a = x["wind"]
    wind_speed = a["speed"]
    wind_speed = np.round(wind_speed*3.6,1)
    wind_dir = a["deg"]
    clouds = x["clouds"]["all"]
    sunrise = int(x["sys"]["sunrise"])+int(x["timezone"])
    sunrise = datetime.datetime.utcfromtimestamp(int(sunrise)).strftime('%H:%M')
    sunset = int(x["sys"]["sunset"])+int(x["timezone"])
    sunset = datetime.datetime.utcfromtimestamp(int(sunset)).strftime('%H:%M')

# Embed  in Image
font = ImageFont.truetype("OpenSans-Regular.ttf", 14)
img = Image.open('wmap.jpg')
img.paste(wicon, (0,0), wicon)
draw = ImageDraw.Draw(img)
draw.text((120, 10),w_desc,(0,0,0),font=font)
tmstr = 'Temp:     '+str(temp)+'°C'
draw.text((120, 24),tmstr,(0,0,0),font=font)
tmstr = 'RH:          '+str(RH)+'%'
draw.text((120, 38),tmstr,(0,0,0),font=font)
tmstr = 'Wind:      '+str(wind_speed)+' km/h from '+str(wind_dir)+'°'
draw.text((120, 52),tmstr,(0,0,0),font=font)
tmstr = 'MSLP:     '+str(mslp)+' hPa'
draw.text((120, 66),tmstr,(0,0,0),font=font)
tmstr = 'Cloud Cover:     '+str(clouds)+'%'
draw.text((120, 80),tmstr,(0,0,0),font=font)
tmstr = 'Sunrise:     '+str(sunrise)
draw.text((120, 94),tmstr,(0,0,0),font=font)
tmstr = 'Sunsey:     '+str(sunset)
draw.text((120, 108),tmstr,(0,0,0),font=font)

st.Image(img, caption='Current Weather: Visakhapatnam')
