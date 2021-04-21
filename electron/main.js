process.env['ELECTRON_DISABLE_SECURITY_WARNINGS'] = 'true';
const {app, BrowserWindow} = require('electron')
const path = require('path')
const contextMenu = require('electron-context-menu');

function createWindow () {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 720,
    minWidth: 1280,
    minHeight: 720,
    frame: false,
    backgroundColor: '#FFF',
    icon: __dirname + '/icon/icon.png',
    webPreferences: {
      nodeIntegration: true,
      enableRemoteModule: true,
      preload: path.join(__dirname, 'preload.js'),
    }
  })
  mainWindow.loadURL('http://localhost:8000/index.html');
}

app.whenReady().then(() => {
  createWindow()
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0)
      createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
    window.close()
  }
})

contextMenu({
});




