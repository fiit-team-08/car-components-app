const remote = require('electron').remote;
const win = remote.BrowserWindow.getFocusedWindow()

document.getElementById('max-button').addEventListener("click", event => {
    if(win.isMaximized())
        win.unmaximize();
    else
        win.maximize();
});

document.getElementById('min-button').addEventListener("click", event => {
    win.minimize();
});

document.getElementById('close-button').addEventListener("click", event => {
    win.close();
});
