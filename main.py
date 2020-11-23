import eel
from analysis.log_file_analyzer import *

eel.init('electron')
eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')


@eel.expose
def getpath(path):
    json_data = get_raw_data_json(path)
    print(json_data)


@eel.expose
def get_track_data(path):
    json = get_track_graph_data(path)
    return json

@eel.expose
def get_laps_data(path):
    json = get_graph_data_json(path)
    return json


data = [69,59,80,81,56,55,40]


@eel.expose
def getdata():
    return data


eel.start('index.html', mode='electron', cmdline_args=['electron'])



