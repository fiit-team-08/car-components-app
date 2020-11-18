const remote = nodeRequire('electron').remote;
const dialog = nodeRequire('electron').remote.dialog;
const win = remote.BrowserWindow.getFocusedWindow();

let file1 = false;
let file2 = false;
let data = [];

eel.getdata()().then((r) => {
    data = r;
});

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
            display: false,
            labels: {
                fontColor: '#000000',
                fontSize: 15
            }
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
        "datasets":[{"label":"Namerané dáta","data":[1,2,3,4,5,6,7],"fill":false,"borderColor":"#254053e6","lineTension":0.1}, {"label":"Simulované dáta","data":[82,54,43,65,56,55,70],"fill":false,"borderColor":"rgba(107,151,177,0.9)","lineTension":0.1}]
    },
    "options": {
        legend: {
            display: true,
            labels: {
                fontColor: '#000000',
                fontSize: 15
            }
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
            display: true,
            labels: {
                fontColor: '#000000',
                fontSize: 15
            }
        }
    }
});

//win.maximize();

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

document.getElementById('komponent').addEventListener("click", event => {
    document.getElementById('w0').style.display = "none";
    document.getElementById('w1').style.display = "block";
    document.getElementById('w2').style.display = "none";
});

document.getElementById('draha').addEventListener("click", event => {
    document.getElementById('name').innerHTML = "Dráha";
    document.getElementById('w0').style.display = "none";
    document.getElementById('w1').style.display = "none";
    document.getElementById('w2').style.display = "block";
    document.getElementById('ww1').style.display = "block";
    document.getElementById('ww2').style.display = "none";
});


document.getElementById('open-button0').addEventListener("click", event => {
    dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [
            { name: 'Log File (.log)', extensions: ['log'] },
        ]
    }).then(result => {
        if (!result.canceled) {
            document.getElementById('file-name0').innerHTML = result.filePaths.toString().split(/(.*)\\/)[2].split(/\.log$/)[0];
            eel.getpath(result.filePaths.toString());
        }
    }).catch(err => {
        console.log(err)
    })
});

document.getElementById('open-button1').addEventListener("click", event => {
    dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [
            { name: 'Log File (.log)', extensions: ['log'] },
        ]
    }).then(result => {
        if (!result.canceled) {
            document.getElementById('file-name1').innerHTML = result.filePaths.toString().split(/(.*)\\/)[2].split(/\.log$/)[0];
        }
    }).catch(err => {
        console.log(err)
    })
});

document.getElementById('open-button2').addEventListener("click", event => {
    dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [
            { name: 'Log File (.log)', extensions: ['log'] },
        ]
    }).then(result => {
        if (!result.canceled) {
            document.getElementById('file-name2').innerHTML = result.filePaths.toString().split(/(.*)\\/)[2].split(/\.log$/)[0];
        }
    }).catch(err => {
        console.log(err)
    })
});

document.getElementById('back-button').addEventListener("click", event => {
    document.getElementById('w0').style.display = "block";
    document.getElementById('w1').style.display = "none";
    document.getElementById('w2').style.display = "none";
});

document.getElementById('c1').addEventListener("click", event => {
    document.getElementById('name').innerHTML = "Koleso";
    document.getElementById('w0').style.display = "none";
    document.getElementById('w1').style.display = "none";
    document.getElementById('w2').style.display = "block";
    document.getElementById('ww1').style.display = "none";
    document.getElementById('ww2').style.display = "block";
    lines2.data.datasets[0].data = data;
    lines2.update();
});

document.getElementById('c2').addEventListener("click", event => {
    document.getElementById('name').innerHTML = "Volant";
    document.getElementById('w0').style.display = "none";
    document.getElementById('w1').style.display = "none";
    document.getElementById('w2').style.display = "block";
    document.getElementById('ww1').style.display = "none";
    document.getElementById('ww2').style.display = "block";
    lines2.update();
});

document.getElementById('c3').addEventListener("click", event => {
    document.getElementById('name').innerHTML = "Brzdy";
    document.getElementById('w0').style.display = "none";
    document.getElementById('w1').style.display = "none";
    document.getElementById('w2').style.display = "block";
    document.getElementById('ww1').style.display = "none";
    document.getElementById('ww2').style.display = "block";
    lines2.update();
});

document.getElementById('print-button').addEventListener("click", event => {
    if (window.getComputedStyle(document.getElementById('ww2')).display === "block") {
        lines2.canvas.parentNode.style.width = '185mm';
        lines3.canvas.parentNode.style.width = '185mm';
        Chart.instances[1].resize();
        Chart.instances[2].resize();
        window.print();
        lines2.canvas.parentNode.style.width = '98%';
        lines3.canvas.parentNode.style.width = '98%';
        Chart.instances[1].resize();
        Chart.instances[2].resize();
    }
    else {
        lines1.canvas.parentNode.style.width = '185mm';
        Chart.instances[0].resize();
        window.print();
        lines1.canvas.parentNode.style.width = '98%';
        Chart.instances[0].resize();
    }
});
