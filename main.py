import eel
from mpc.mpc import mpc
from mpc.mpc import get_reference_data

from sys import platform
from command_prediction.command_prediction_simple import *
from animation import animation_rendering

eel.init('electron')

if platform == 'darwin': # MacOS
    eel.browsers.set_path('electron', 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
else:
    eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')

laps = None
mpc_data = None
scp_data = None

car_dimensions = []

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
    json = get_reference_xy(get_reference_data(path))
    return json


@eel.expose
def get_mpc_ref_crs(path):
    json = get_reference_crs(get_reference_data(path))
    return json


@eel.expose
def get_mpc_xy(path, length, width, backtowheel, wb, wheel, target_speed, max_speed, max_accel, max_steer, max_dsteer):
    global mpc_data, car_dimensions
    car_dimensions = [length, width, backtowheel, wb, wheel]
    mpc_data = mpc(path, length, width, backtowheel, wb, wheel, target_speed, max_speed, max_accel, max_steer, max_dsteer)
    temp = mpc_data.copy()
    json = get_data_xy(temp)
    return json


@eel.expose
def get_mpc_crs(referenceFileName):
    temp = mpc_data.copy()
    json = get_data_crs(temp)
    return json


# SIMPLE COMMAND PREDICION

@eel.expose
def get_scp_xy(reference_path, traces_path):
    global scp_data, car_dimensions
    car_dimensions = []
    scp_data = get_simple_command_prediction(reference_path, traces_path)
    temp = scp_data.copy()
    json = get_data_xy(temp)
    return json


@eel.expose
def get_scp_crs():
    temp = scp_data.copy()
    json = get_data_crs(temp)
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


# ANIMATION
@eel.expose
def animate_track(model):
    if model == 'scp':
        data = rename_columns(scp_data.copy())
    elif model == 'mpc':
        data = rename_columns(mpc_data.copy())
    animation_rendering.run_animation(data, car_dimensions)


data = [69, 59, 80, 81, 56, 55, 40]


@eel.expose
def getdata():
    return data


eel.start('index.html', mode='electron', cmdline_args=['electron'])
