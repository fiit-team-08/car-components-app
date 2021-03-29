import eel
from mpc.mpc import *
from analysis.log_file_analyzer import *


eel.init('electron')
eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')

laps = None
mpc_data = None

@eel.expose
def getpath(path):
    json_data = get_raw_data_json(path)
    print(json_data)


@eel.expose
def get_track_data(path):
    json = get_track_graph_data(path)
    return json


@eel.expose
def get_mpc_ref_xy(path):
    json = get_mpc_reference_xy(get_reference_data(path))
    return json


@eel.expose
def get_mpc_ref_crs(path):
    json = get_mpc_reference_crs(get_reference_data(path))
    return json


@eel.expose
def get_mpc_xy(path, length, width, backtowheel, wb, wheel, target_speed, max_speed, max_accel, max_steer, max_dsteer):
    global mpc_data
    mpc_data = mpc(path, length, width, backtowheel, wb, wheel, target_speed, max_speed, max_accel, max_steer, max_dsteer)
    temp = mpc_data.copy()
    json = get_mpc_data_xy(temp)
    return json


@eel.expose
def get_mpc_crs(referenceFileName):
    temp = mpc_data.copy()
    json = get_mpc_data_crs(temp)
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


@eel.expose
def export_mpc(path, file_name):
    if mpc_data is None:
        return
    mpc_data.to_csv(path+'\out.csv', index=False)


@eel.expose
def get_reference_laps(path):
    return get_number_of_lines(path)


data = [69, 59, 80, 81, 56, 55, 40]


@eel.expose
def getdata():
    return data


eel.start('index.html', mode='electron', cmdline_args=['electron'])
