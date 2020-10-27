const {app, BrowserWindow, Menu, ipcMain} = require('electron')
const path = require('path')
const ipc = require('electron').ipcMain;
const contextMenu = require('electron-context-menu');

app.commandLine.appendSwitch('remote-debugging-port', '9222')

function createWindow () {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 720,
    frame: false,
    backgroundColor: '#FFF',
    icon: __dirname + '/icons/icon.png',
    webPreferences: {
      nodeIntegration: true,
      enableRemoteModule: true,
      preload: path.join(__dirname, 'preload.js'),
    }
  })
  mainWindow.loadFile('index.html')
}

app.whenReady().then(() => {
  createWindow()
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})

ipc.on('invokeAction', function(event, data){
  const result = processData(data);
  event.sender.send('actionReply', result);
});

contextMenu({
});


