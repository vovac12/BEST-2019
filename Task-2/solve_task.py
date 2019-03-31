#!/usr/local/bin/python

import pandas as pd
import numpy as np
import json
import scipy
import getpass
import socket
import random
import math
from matplotlib import pyplot as plt
import sys

TEAM_KEY = sys.argv[1]
TASK_NUM = int(sys.argv[2])
print(TASK_NUM)

def bytelen(obj):
    if type(obj) == str:
        obj = obj.encode()
    size = len(bytes(obj))
    return bytes((size % (256 ** i) // 256**(i-1) for i in range(1, 5)))

def byte_count(x):
    x = list(x)
    return sum((x[i] * 256**i for i in range(len(x))))

def connect_to_server(url, port):
    sock = socket.socket()
    sock.connect((url, port))
    return sock

def send_data(sock, data):
    if type(data) == str:
        data = data.encode()
    data = bytes(data)
    if type(sock) == socket.socket:
        return sock.send(bytelen(data) + data)

def recv_data(sock):
    if type(sock) == socket.socket:
        reader = sock.recv
        buff = 1000000
    else:
        reader = sock.read
        buff = 100000
    l = reader(4)
    l = byte_count(l)
    res = []
    cnt = 0
    while cnt < l:
        tmp = list(reader(buff if l - cnt > buff else l - cnt))
        cnt += len(tmp)
        res.extend(tmp)
    res = bytes(res)
    if not res:
        return None
    return json.loads(res)

def dict_to_bytes(data):
    return json.dumps(data).encode()

def get_start_msg(task):
    return dict_to_bytes({'team': TEAM_KEY, 'task': task})

def send_result(sock, x, y, ready):
    if type(sock) != socket.socket:
        return 0
    data = {"x": x, "y":y, "ready": ready}
    return send_data(sock, dict_to_bytes(data))

def get_map(sock):
    return json.loads(recv_data(sock))

hello_content = json.dumps({'team': TEAM_KEY, 'task': TASK_NUM}).encode()
TIME_OUT = 20

def find_sollution(landMap, hs, currentData):
    if (hs is None):
        #print(landMap[:1] - np.full(landMap[0].size, currentData['height']))
        heights = np.vstack([abs(landMap[:1] - np.full(landMap[0].size, currentData['height'])), landMap[1:]])
        heights = (heights.T[heights[0].argsort()].T)[:, :2000]
        #print("BEGIN", heights)
        #print("BEGIN", heights.shape)
    else:
        heights = hs.copy()
        heights -= np.vstack([np.full(heights[0].size, currentData['height']), np.zeros((2, heights[0].size), "int64")])
        heights = abs(heights)
        heights = heights.T[heights[0].argsort()].T
    #if heights[0][0] > 30000:
        #heights = np.vstack([abs(landMap[:1] - np.full(landMap[0].size, currentData['height'])), landMap[1:]])
        #heights = (heights.T[heights[0].argsort()].T)[:, :300]
    res = [heights[1][0], heights[2][0]]
    #print("RES : ", res, "EXPECTED : ", currentData['x'], currentData['y'])
    ans = predict_height(heights[1:2], heights[2:],currentData['speed'], currentData['psi'])
    edge = int(math.sqrt(landMap[0][1]))
    #heights[1:2][heights[1:2] > edge] = edge
    #heights[1:][heights[1:] < 0] = 0
    heights[0] = ans[2]
    return [res, heights]

ideal_route_x = []
ideal_route_y = []
my_route_x = []
my_route_y = []
predicted_x = []
predicted_y = []
ideal_height = []
my_height = []

def predict_height(x, y, speed, psi):
    COF = 1
    x += round(COF * speed * math.sin(math.radians(psi)))
    y += round(COF * speed * math.cos(-math.radians(psi)))
    x = np.around(x).astype(int)
    y = np.around(y).astype(int)
    edge = int(math.sqrt(landMap[0][1]))
    height = landMap[0][x + y * edge]
    return (x, y, height)

#Communication loop
sock = connect_to_server("besthack19.sytes.net", 4242)
sock.settimeout(TIME_OUT)
#sock = open('/home/kami/git/BEST-2019/Task-2/Data/task' + str(TASK_NUM), 'rb')
send_data(sock, hello_content)
landMap = recv_data(sock)
if landMap.get('error'):
    print(str(TASK_NUM) + " ERROR: ", landMap["error"])
    exit()
landMap = landMap["map"]
mapLen = int(math.sqrt(len(landMap)))
landMap = np.array(landMap)
lMmin = landMap.min()
lMmax = landMap.max()
if lMmax == lMmin:
    lMmax += 1
#landMap = (landMap - lMmin) / (lMmax - lMmin)
#print(mapLen)
landMap = np.vstack([np.array(landMap), np.array([i % mapLen for i in range(0, landMap.size)]), np.array([i // mapLen for i in range(0, landMap.size)])])
heights = None
cnt = 0
p_coor = None
inert = np.array((0, 0))

while (True):
    print(cnt)
    data = recv_data(sock)
    if data.get('error'):
        print(str(TASK_NUM) + "ERROR: ", data["error"])
        break
    if data.get('scores'):
        print(str(TASK_NUM) + "SCORE ",data['scores'])
        break
    if not data['data']['speed']:
        data['data']['speed'] = 2
    #data['data']['height'] = (data['data']['height'] - lMmin) / (lMmax - lMmin)
    res = find_sollution(landMap, heights, data['data'])

    heights = res[1]
    if res[0][0] > mapLen:
        res[0][0] = mapLen
    if res[0][1] > mapLen:
        res[0][1] = mapLen
    if res[0][0] < 0:
        res[0][0] = 0
    if res[0][1] < 0:
        res[0][1] = 0
    cnt += 1
    cur_x = res[0][0]
    cur_y = res[0][1]
    max_speed = data['data']['speed'] * 1.5 if data['data']['speed'] else 3
    if p_coor is not None:
        cdelta = np.array([cur_x - p_coor[0], cur_y - p_coor[1]])
        if np.linalg.norm(cdelta) > max_speed:
            cdelta = cdelta / np.linalg.norm(cdelta) * max_speed
        inert = cdelta #inert * 0.5 + cdelta
        cur_x = p_coor[0] + inert[0]
        cur_y = p_coor[1] + inert[1]

    p_coor = (cur_x, cur_y)
    send_result(sock, int(round(cur_x)), int(round(cur_y)), 1)