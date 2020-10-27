const remote = require('electron').remote;
const dialog = require('electron').remote.dialog;
const win = remote.BrowserWindow.getFocusedWindow();

let file = false;
let component = 0;

let d = [15,59,80,81,56,55,40];
let dd = [65,59,80,81,56,55,40];

let lines = new Chart(document.getElementById("chart"), {
    "type":"line","data": {
        "labels":["1","2","3","4","5","6","7"],
        "datasets":[{"label":"Namerané dáta","data":d,"fill":false,"borderColor":"#254053e6","lineTension":0.1}, {"label":"Simulované dáta","data":[82,54,43,65,56,55,70],"fill":false,"borderColor":"rgba(107,151,177,0.9)","lineTension":0.1}]
    },
    "options": {
        legend: {
            display: true
        }
    }
});

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

document.getElementById('ok-button').style.background = "#8da0ad";

document.getElementById('open-button').addEventListener("click", event => {
    dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [
            { name: 'Log File (.csv)', extensions: ['csv'] },
        ]
    }).then(result => {
        if (!result.canceled) {
            document.getElementById('file-name').innerHTML = result.filePaths.toString().split(/(.*)\\/)[2].split(/\.csv$/)[0];
            file = true;
            if (component > 0) {
                document.getElementById('ok-button').style.background = null;
                document.getElementById('ok-button').style.cursor = "pointer";
                document.getElementById('ok-button').style.transition = "0.5s ease-in-out";
            }
        }
    }).catch(err => {
        console.log(err)
    })
});

document.getElementById('c1').addEventListener("click", event => {
    document.getElementById('c1').style.background = "#dddddd";
    document.getElementById('c2').style.background = null;
    document.getElementById('c3').style.background = null;
    document.getElementById('name').innerHTML = "Koleso";
    component = 1;
    if (file) {
        document.getElementById('ok-button').style.background = null;
        document.getElementById('ok-button').style.cursor = "pointer";
        document.getElementById('ok-button').style.transition = "0.5s ease-in-out";
    }
});

document.getElementById('c2').addEventListener("click", event => {
    document.getElementById('c1').style.background = null;
    document.getElementById('c2').style.background = "#dddddd";
    document.getElementById('c3').style.background = null;
    document.getElementById('name').innerHTML = "Volant";
    component = 1;
    if (file) {
        document.getElementById('ok-button').style.background = null;
        document.getElementById('ok-button').style.cursor = "pointer";
        document.getElementById('ok-button').style.transition = "0.5s ease-in-out";
    }
});

document.getElementById('c3').addEventListener("click", event => {
    document.getElementById('c1').style.background = null;
    document.getElementById('c2').style.background = null;
    document.getElementById('c3').style.background = "#dddddd";
    document.getElementById('name').innerHTML = "Brzdy";
    component = 1;
    if (file) {
        document.getElementById('ok-button').style.background = null;
        document.getElementById('ok-button').style.cursor = "pointer";
        document.getElementById('ok-button').style.transition = "0.5s ease-in-out";
    }
});

document.getElementById('ok-button').addEventListener("click", event => {
    if (component > 0 && file) {
        document.getElementById('w1').style.display = "none";
        document.getElementById('w2').style.display = "block";
        if(Math.floor(Math.random() * 11) > 4)
            lines.data.datasets[0].data = dd;
        else
            lines.data.datasets[0].data = d;
        lines.update();
    }
});

document.getElementById('back-button').addEventListener("click", event => {
    if (component > 0 && file) {
        document.getElementById('w1').style.display = "block";
        document.getElementById('w2').style.display = "none";
    }
});