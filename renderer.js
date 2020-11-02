const remote = require('electron').remote;
const dialog = require('electron').remote.dialog;
const win = remote.BrowserWindow.getFocusedWindow();

let file = false;
let component = 0;

let lines1 = new Chart(document.getElementById("chart1"),{
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Dráha',
            showLine: true,
            fill: false,
            "borderColor":"#254053e6",
            data: [{
                x: 0,
                y: -4
            }, {
                x: 5,
                y: 0
            }, {
                x: 0,
                y: 0
            }, {
                x: 5,
                y: 5
            }, {
                x: 0,
                y: 6
            }, {
                x: -5,
                y: 0
            }, {
                x: -3,
                y: -2
            }, {
                x: 0,
                y: -4
            }]
        }]
    },
    options: {
        legend: {
            display: false
        },
        tooltips: {
            enabled: false
        },
        elements: {
            point:{
                radius: 0
            }
        },
        scales: {
            xAxes: [{
                gridLines: {
                    display:false
                },
                ticks: {
                    fontColor: "white"
                }
            }],
            yAxes: [{
                gridLines: {
                    display:false
                },
                ticks: {
                    fontColor: "white"
                }
            }]
        }
    }
});

let lines2 = new Chart(document.getElementById("chart2"), {
    "type":"line","data": {
        "labels":["1","2","3","4","5","6","7"],
        "datasets":[{"label":"Namerané dáta","data":[15,59,80,81,56,55,40],"fill":false,"borderColor":"#254053e6","lineTension":0.1}, {"label":"Simulované dáta","data":[82,54,43,65,56,55,70],"fill":false,"borderColor":"rgba(107,151,177,0.9)","lineTension":0.1}]
    },
    "options": {
        legend: {
            display: true
        }
    }
});

let lines3 = new Chart(document.getElementById("chart3"), {
    "type":"line","data": {
        "labels":["1","2","3","4","5","6","7"],
        "datasets":[{"label":"Rozdielne dáta","data":[82,54,43,65,56,55,70],"fill":false,"borderColor":"rgba(107,151,177,0.9)","lineTension":0.1}]
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
            { name: 'Log File (.log)', extensions: ['log'] },
        ]
    }).then(result => {
        if (!result.canceled) {
            document.getElementById('file-name').innerHTML = result.filePaths.toString().split(/(.*)\\/)[2].split(/\.log$/)[0];
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
        //lines1.data.datasets[0].data = d;
        //lines1.update();
    }
});

document.getElementById('back-button').addEventListener("click", event => {
    if (component > 0 && file) {
        document.getElementById('w1').style.display = "block";
        document.getElementById('w2').style.display = "none";
    }
});

document.getElementById('print-button').addEventListener("click", event => {
    lines1.canvas.parentNode.style.width = '185mm';
    lines2.canvas.parentNode.style.width = '185mm';
    lines3.canvas.parentNode.style.width = '185mm';
    for (let id in Chart.instances) {
        Chart.instances[id].resize();
    }
    window.print();
    lines1.canvas.parentNode.style.width = '95%';
    lines2.canvas.parentNode.style.width = '95%';
    lines3.canvas.parentNode.style.width = '95%';
    for (let id in Chart.instances) {
        Chart.instances[id].resize();
    }
});