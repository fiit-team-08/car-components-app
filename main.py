import eel
from analysis.log_file_analyzer import *

eel.init('electron')
eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')

laps = None


@eel.expose
def getpath(path):
    json_data = get_raw_data_json(path)
    print(json_data)


@eel.expose
def get_track_data(path):
    json = get_track_graph_data(path)
    return json


@eel.expose
def get_laps_data(reference_file_path, traces_file_path):
    global laps
    laps = get_lap_data(reference_file_path, traces_file_path)
    json = put_laps_to_json(laps)
    return json


@eel.expose
def export_data(path, file_name):
    print("Exporting to: {}\{}".format(path, file_name))
    if laps is None:
        return
    save_laps_to_files(path, file_name, laps)


data = [69, 59, 80, 81, 56, 55, 40]


@eel.expose
def getdata():
    return data


eel.start('index.html', mode='electron', cmdline_args=['electron'])
