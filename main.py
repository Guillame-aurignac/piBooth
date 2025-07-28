# Raspberry pi 5 based photo booth
# Guillaume AURIGNAC
# 2025
# v1.0.1

import copy
import cv2
import json
import numpy as np
import os
from pprint import *
import random
import requests
import gpiod # Import Raspberry Pi 5 GPIO library
import time
from collections import namedtuple
from datetime import datetime
from dotenv import load_dotenv, dotenv_values
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont, ImageChops
from screeninfo import get_monitors


_ntuple_diskusage = namedtuple('usage', 'total used free per_used')
def disk_usage(path):
    """Return disk usage statistics about the given path.

    Returned valus is a named tuple with attributes 'total', 'used' and
    'free', which are the amount of total, used and free space, in bytes.
    """
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    per_used = round(used*100/total,2)
    return _ntuple_diskusage(total, used, free, per_used)

USE_IMMICH = True
MOVE_TO_ALBUM = True
GPS_INFO = True

FILENAME_BASE = "booth_"
FILENAME_INDEX = len(os.listdir("photos/"))+1

# loading variables from .env file
# load_dotenv()
config = dotenv_values(".env")

k = config.keys()

if not "API_KEY" in k:
    print("Missing API key from env file, skipping Immich upload")
    USE_IMMICH = False

if not "BASE_URL" in k:
    print("Missing server url from env file, skipping Immich upload")
    USE_IMMICH = False

if not "ALBUM_ID" in k:
    print("Missing album ID from env file, skipping move to album")
    MOVE_TO_ALBUM = False

if not "latitude" in k:
    print("Missing latitude from env file, skipping position information")
    GPS_INFO = False
if not "longitude" in k:
    print("Missing longitude from env file, skipping position information")
    GPS_INFO = False

sepia_filter = np.array([[.272, .534, .131],
                        [.349, .686, .168],
                        [.393, .769, .189]])

def sepia (image):
    # here goes the filtering
    sepia_img = image.dot(sepia_filter.T)

    # Unfortunately your filter lines do not have unit sum, so we need to rescale
    sepia_img /= sepia_img.max()
    sepia_img *= 255
    #print(sepia_img[0][0])
    #exit()

    return sepia_img.astype(np.uint8)

def effect(image):    
    # list all available effects
    effects_file = random.choice(os.listdir("effects/"))

    # convert to PIL
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    pil_im =Image.fromarray(rgb)

    with Image.open("effects/"+effects_file) as pil_effect:
        #if effects_file == "booth_effect_1.png":
        #    pil_im = ImageChops.multiply(pil_effect, pil_im)
        #else:
        pil_im.paste(pil_effect, (0, 0), pil_effect)
    
    # convert back to cv2 image
    rgb = np.array(pil_im)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return image

def upload(file):
    stats = os.stat(file)

    headers = {
        'Accept': 'application/json',
        'x-api-key': config["API_KEY"]
    }

    data = {
        'deviceAssetId': f'{file}-{stats.st_mtime}',
        'deviceId': 'PhotoBooth',
        'fileCreatedAt': datetime.fromtimestamp(stats.st_mtime),
        'fileModifiedAt': datetime.fromtimestamp(stats.st_mtime)
    }

    files = {
        'assetData': open(file, 'rb')
    }

    response = requests.post(f'{config["BASE_URL"]}/assets', headers=headers, data=data, files=files)

    status = response.json()['status']
    asset_id = response.json()['id']

    if status == "created":
        # Add to album
        if MOVE_TO_ALBUM:
            url = f"{config['BASE_URL']}/albums/{config['ALBUM_ID']}/assets"
            payload = json.dumps({"ids": [asset_id]})
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'x-api-key': config["API_KEY"]
            }

            _ = requests.request("PUT", url, headers=headers, data=payload)

        # Update asset with localisation
        if GPS_INFO:
            url = f"{config['BASE_URL']}/assets"
            payload = json.dumps({
                "ids": [asset_id],
                "latitude": float(config["latitude"]),
                "longitude": float(config['longitude'])
            })
            
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': config["API_KEY"]
            }

            _ = requests.request("PUT", url, headers=headers, data=payload)


monitor = get_monitors()[0]
camera_param = {"width": 1456, "height": 1088}
#camera_param = {"width": 640, "height": 480}

delta_width = monitor.width - camera_param['width']
delta_height = monitor.height - camera_param['height']

if delta_width < delta_height:
    # resizing by width
    new_width = monitor.width
    new_height = int((monitor.width/camera_param['width']) * camera_param['height'])

elif delta_width > delta_height:
    # resizing by height
    new_width = int((monitor.height/camera_param['height']) * camera_param['width'])
    new_height = monitor.height
else:
    # no resizing
    new_width = camera_param['width']
    new_height = camera_param['height']

x_offset = int((monitor.width-new_width)/2)
y_offset = int((monitor.height-new_height)/2)

background = np.full((monitor.height, monitor.width, 3), 0, dtype = np.uint8)
flash = np.full((monitor.height, monitor.width, 4), 255, dtype = np.uint8)

cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (camera_param["width"], camera_param["height"])}))
picam2.start()

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Define GPIO
BUTTON_PIN = 27
chip = gpiod.Chip('gpiochip4')
button_line = chip.get_line(BUTTON_PIN)
button_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)

# Define font for HMI
font_title = ImageFont.truetype("fonts/Carnevalee Freakshow.ttf", 250)
font_subtitle = ImageFont.truetype("fonts/Carnevalee Freakshow.ttf", 120)
font_number = ImageFont.truetype("fonts/Carnevalee Freakshow.ttf", 1000)
font_debug = ImageFont.truetype("fonts/arial.ttf", 20)

# DEBUG mode to display debug info
DEBUG = False
# reverse the image
REVERSED = True

inCountDown = False
count_down_timeout = 6 # in seconds
t_countdown = 0
count_down_value = 0

inIdle = True
idle_timeout = 15 # in seconds

displayLast = False
last_timeout = 5

# CPU temp monitooring vars
temp_check_interval = 1 # in seconds
t_prev_temp = 0
cpu_temp = os.popen('vcgencmd measure_temp').readline()

# Network check vars
network_check_interval = 60 # in seconds
t_prev_net = 0

# Disk space left monitoring
# at interval to catch low disk space if other
# process are writing
# and every time image is saved
disk_check_interval = 5*60 # in seconds
t_prev_disk = 0
disk_used_space = disk_usage('.').per_used

t_prev = time.time()
t_prev_idle = 0
t_prev_last = 0

while True:
    if displayLast:
        im = im_filter
    else:
        im = picam2.capture_array()
        # remove alpha layer
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

        if REVERSED:
            # flip horizontaly
            im = cv2.flip(im, 1)

    # resize camera
    resized = cv2.resize(copy.copy(im),(new_width,new_height))    
    # paste over background
    background[y_offset:new_height+y_offset, x_offset:new_width+x_offset] = resized

    # compute fps
    fps = str(round(1/(time.time() - t_prev)))

    # convert to PIL
    rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    pil_im =Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_im)

    if DEBUG:
        # position text at 2% of the monitor width to the right edge
        x = monitor.width - int(monitor.width*0.02)

        texts = [f"fps: {fps}",
                 f"Photo count: {FILENAME_INDEX-1}",
                 f"Reversed: {REVERSED}",
                 cpu_temp,
                 f"Disk used: {disk_used_space}%"]

        # print debug info
        for i in range(len(texts)):
            draw.text((x,70+30*i), texts[i], fill=(255,0,0), font=font_debug, anchor="rs")

    if inCountDown:
        org = int(monitor.width/2),int(monitor.height/2)
        count_down_value = int(count_down_timeout - (time.time() - t_countdown))

        if count_down_value:
            draw.text(org, str(count_down_value), fill=(255,252,226), font=font_number, anchor="mm", stroke_width=1, stroke_fill=(0,0,0))

        if count_down_timeout - (time.time() - t_countdown) <= 0:
            # take picture
            filename = f"photos/{FILENAME_BASE}{FILENAME_INDEX}.jpg"
            FILENAME_INDEX += 1

            # sepia filter
            im_filter = sepia(im)
            # add effect from file
            im_filter = effect(im_filter)

            # save image to disk
            cv2.imwrite(filename, im_filter)
            # update disk space used
            disk_used_space = disk_usage('.').per_used

            if USE_IMMICH:
                upload(filename)

            inCountDown = False
            displayLast = True
            t_prev_last = time.time()

    inIdle = (time.time() - t_prev_idle > idle_timeout)
    displayLast = (time.time() - t_prev_last <= last_timeout)

    if inIdle:
        org = int(monitor.width/2),int(monitor.height/2)
        draw.text(org, "Photo Shoot", fill=(255,252,226), font=font_title, anchor="md", stroke_width=1, stroke_fill=(0,0,0))
        draw.text(org, "Press to start", fill=(255,252,226), font=font_subtitle, anchor="ma", stroke_width=1, stroke_fill=(0,0,0))

    t_prev = time.time()

    # convert back to cv2 image
    rgb = np.array(pil_im)
    im_osd = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if inCountDown and count_down_value == 0:
        cv2.imshow('window', flash)
    else:
        cv2.imshow("window", im_osd)
    
    k = cv2.waitKey(1)
    if  k == 27 or k == ord('q'): #Esc
        cv2.destroyAllWindows()
        break

    elif not displayLast and not inCountDown and (k == 32 or button_line.get_value()): # space or button press
       t_countdown = time.time()
       inCountDown = True
       t_prev_idle = time.time()
       inIdle = False

    elif k == ord('d'):
        DEBUG = not DEBUG

    elif k == ord('r'):
        REVERSED = not REVERSED

    # === Timers ===
    if (time.time() - t_prev_temp) > temp_check_interval:
        cpu_temp = os.popen('vcgencmd measure_temp').readline()
        t_prev_temp = time.time()

    if (time.time() - t_prev_disk) > disk_check_interval:
        disk_used_space = disk_usage('.').per_used
        t_prev_temp = time.time()