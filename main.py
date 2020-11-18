import eel
from analysis.log_file_analyzer import *

eel.init('electron')
eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')


@eel.expose
def getpath(path):
    print(normalize_logs(log_to_dataFrame(path)))


data = [69,59,80,81,56,55,40]


@eel.expose
def getdata():
    return data


eel.start('index.html', mode='electron', cmdline_args=['electron'])



