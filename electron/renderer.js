const remote = nodeRequire('electron').remote;
const dialog = nodeRequire('electron').remote.dialog;
const win = remote.BrowserWindow.getFocusedWindow();

let file1 = 0;
let file2 = 0;
let sum = 0;
let data = [];
let trackdata = undefined;
let referenceFileName = undefined;
let tracesFileName = undefined;
let selector = 1;
let car = 0;
let drahadesc = "";
let vozidlodesc = "";


eel.getdata()().then((r) => {
    data = r;
});

//win.maximize();

let lines1 = new Chart(document.getElementById("chart1"), {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Referenčná dráha',
            showLine: true,
            fill: false,
            "borderColor": "#B71C1C",
            borderWidth: 5,
            data: []
        }]
    },
    options: {
        aspectRatio: 1,
        legend: {
            display: false,
            labels: {
                fontColor: '#000000',
                fontSize: 15
            }
        },
        layout: {
            padding: 5
        },
        tooltips: {
            enabled: false
        },
        elements: {
            point: {
                radius: 0
            }
        },
        scales: {
            xAxes: [{
                gridLines: {
                    display: false
                },
                ticks: {
                    display: false,
                    fontColor: "#f5f5f5",
                    stepSize: 1,
                    stepValue: 1
                }
            }],
            yAxes: [{
                gridLines: {
                    display: false
                },
                ticks: {
                    display: false,
                    fontColor: "#f5f5f5",
                    stepSize: 1,
                    stepValue: 1
                }
            }]
        }
    }
});

let lines2 = new Chart(document.getElementById("chart2"), {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Referenčná dráha',
            showLine: true,
            fill: false,
            "borderColor": "#B71C1C",
            borderWidth: 5,
            data: []
        },
            {
                label: 'SCP',
                showLine: true,
                fill: false,
                "borderColor": "#4e7490",
                data: []
            },
            {
                label: 'MPC',
                showLine: true,
                fill: false,
                "borderColor": "#a1bacd",
                data: []
            }]
    },
    options: {
        aspectRatio: 1,
        legend: {
            display: false,
            labels: {
                fontColor: '#000000',
                fontSize: 15
            }
        },
        layout: {
            padding: 5
        },
        tooltips: {
            enabled: false
        },
        elements: {
            point: {
                radius: 0
            }
        },
        scales: {
            xAxes: [{
                gridLines: {
                    display: false
                },
                ticks: {
                    display: false,
                    fontColor: "#f5f5f5",
                    stepSize: 1,
                    stepValue: 1
                }
            }],
            yAxes: [{
                gridLines: {
                    display: false
                },
                ticks: {
                    display: false,
                    fontColor: "#f5f5f5",
                    stepSize: 1,
                    stepValue: 1
                }
            }]
        }
    }
});

let lines3 = new Chart(document.getElementById("chart3"), {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Referenčná dráha',
            showLine: true,
            fill: false,
            "borderColor": "#B71C1C",
            borderWidth: 5,
            data: []
        },
            {
                label: 'SCP',
                showLine: true,
                fill: false,
                "borderColor": "#4e7490",
                data: []
            },
            {
                label: 'MPC',
                showLine: true,
                fill: false,
                "borderColor": "#a1bacd",
                data: []
            }]
    },
    options: {
        legend: {
            display: false,
            labels: {
                fontSize: 0
            }
        },
        tooltips: {
            enabled: false
        },
        elements: {
            point: {
                radius: 0
            }
        },
        scales: {
            xAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: 'Čas'
                }
            }],
            yAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: 'Natočenie auta'
                }
            }]
        }
    }
});

function loadChartForFile(name, index) {
    eel.get_track_data(name)().then((r) => {
        if (index === 0) {
            lines1.data.datasets[index].data = JSON.parse(r);
            lines1.update()
        }
    });
}

function loadReference(name) {
    eel.get_mpc_ref_xy(name)().then((r) => {
        lines2.data.datasets[0].data = JSON.parse(r);
        lines2.update()
    });
    eel.get_mpc_ref_crs(name)().then((r) => {
        lines3.data.datasets[0].data = JSON.parse(r);
        lines3.update()
    });
}

function lines2ref() {
    if (lines2.data.datasets[0].hidden === true)
        lines2.data.datasets[0].hidden = false;
    else
        lines2.data.datasets[0].hidden = true;
    lines2.update()
}

function lines3ref() {
    if (lines3.data.datasets[0].hidden === true)
        lines3.data.datasets[0].hidden = false;
    else
        lines3.data.datasets[0].hidden = true;
    lines3.update()
}

function lines2scp() {
    if (lines2.data.datasets[1].hidden === true)
        lines2.data.datasets[1].hidden = false;
    else
        lines2.data.datasets[1].hidden = true;
    lines2.update()
}

function lines3scp() {
    if (lines3.data.datasets[1].hidden === true)
        lines3.data.datasets[1].hidden = false;
    else
        lines3.data.datasets[1].hidden = true;
    lines3.update()
}

function lines2mpc() {
    if (lines2.data.datasets[2].hidden === true)
        lines2.data.datasets[2].hidden = false;
    else
        lines2.data.datasets[2].hidden = true;
    lines2.update()
}

function lines3mpc() {
    if (lines3.data.datasets[2].hidden === true)
        lines3.data.datasets[2].hidden = false;
    else
        lines3.data.datasets[2].hidden = true;
    lines3.update()
}

function runcloud() {
    document.getElementById('loading').style.display = "flex";
    document.getElementsByClassName('disable')[0].style.opacity = "0.5";
    document.getElementsByClassName('disable')[0].style.pointerEvents = "none";
    document.getElementsByClassName('disable')[1].style.opacity = "0.5";
    document.getElementsByClassName('disable')[1].style.pointerEvents = "none";
    document.getElementById('loading').style.display = "none";
}

function runlocal() {
    document.getElementById('loading').style.display = "flex";
    document.getElementsByClassName('disable')[0].style.opacity = "0.5";
    document.getElementsByClassName('disable')[0].style.pointerEvents = "none";
    document.getElementsByClassName('disable')[1].style.opacity = "0.5";
    document.getElementsByClassName('disable')[1].style.pointerEvents = "none";
    if (selector === 1) {
        eel.get_scp_xy(referenceFileName, tracesFileName)().then((r) => {
            console.log(r)
            lines2.data.datasets[1].data = JSON.parse(r);
            lines2.update()
        });
        eel.get_scp_crs()().then((r) => {
            lines3.data.datasets[1].data = JSON.parse(r);
            lines3.update()

            document.getElementById("trasyxy").appendChild(document.createElement('br'));
            document.getElementById("trasycrs").appendChild(document.createElement('br'));
            let item1 = document.createElement('input');
            item1.setAttribute('id', 'scpxy');
            item1.setAttribute('type', 'checkbox');
            item1.setAttribute('onclick', 'lines2scp()');
            item1.setAttribute('checked', 'true');
            let item2 = document.createElement('input');
            item2.setAttribute('id', 'scpcsr');
            item2.setAttribute('type', 'checkbox');
            item2.setAttribute('onclick', 'lines3scp()');
            item2.setAttribute('checked', 'true');
            document.getElementById("trasyxy").appendChild(item1);
            document.getElementById("trasycrs").appendChild(item2);

            let label1 = document.createElement('label');
            label1.appendChild(document.createTextNode("Simple Command Prediction"));
            label1.setAttribute('for', 'scpxy');
            label1.setAttribute('class', 'scpxy');
            let label2 = document.createElement('label');
            label2.appendChild(document.createTextNode("Simple Command Prediction"));
            label2.setAttribute('for', 'scpcsr');
            label2.setAttribute('class', 'scpcsr');
            document.getElementById("trasyxy").appendChild(label1);
            document.getElementById("trasycrs").appendChild(label2);
            document.getElementsByClassName('disable')[0].style.opacity = "1";
            document.getElementsByClassName('disable')[0].style.pointerEvents = "auto";
            document.getElementsByClassName('disable')[1].style.opacity = "1";
            document.getElementsByClassName('disable')[1].style.pointerEvents = "auto";
            document.getElementsByClassName('run')[0].style.opacity = "1";
            document.getElementsByClassName('run')[0].style.pointerEvents = "auto";
            document.getElementById('loading').style.display = "none";
        });
    }
    if (selector === 2) {
        let lenght
        let width
        let wheels
        let wheelbase
        let wheelsize
        let s
        let speed
        let acceleration
        let angle
        let anglespeed
        if (document.getElementById("lenght").value !== "")
            lenght = parseFloat(document.getElementById("lenght").value.replace(",", "."))
        else
            lenght = 2.817
        if (document.getElementById("width").value !== "")
            width = parseFloat(document.getElementById("width").value.replace(",", "."))
        else
            width = 1.680
        if (document.getElementById("wheels").value !== "")
            wheels = parseFloat(document.getElementById("wheels").value.replace(",", "."))
        else
            wheels = 1.480
        if (document.getElementById("wheelbase").value !== "")
            wheelbase = parseFloat(document.getElementById("wheelbase").value.replace(",", "."))
        else
            wheelbase = 2.345
        if (document.getElementById("wheelsize").value !== "")
            wheelsize = parseFloat(document.getElementById("wheelsize").value.replace(",", "."))
        else
            wheelsize = 0.300
        if (document.getElementById("s").value !== "")
            s = parseFloat(document.getElementById("s").value.replace(",", "."))
        else
            s = 25
        if (document.getElementById("speed").value !== "")
            speed = parseFloat(document.getElementById("speed").value.replace(",", "."))
        else
            speed = 40
        if (document.getElementById("acceleration").value !== "")
            acceleration = parseFloat(document.getElementById("acceleration").value.replace(",", "."))
        else
            acceleration = 1
        if (document.getElementById("angle").value !== "")
            angle = parseFloat(document.getElementById("angle").value.replace(",", "."))
        else
            angle = 30
        if (document.getElementById("anglespeed").value !== "")
            anglespeed = parseFloat(document.getElementById("anglespeed").value.replace(",", "."))
        else
            anglespeed = 10
        eel.get_mpc_xy(referenceFileName, lenght, width, wheels, wheelbase, wheelsize, s, speed, acceleration, angle, anglespeed)().then((r) => {
            lines2.data.datasets[2].data = JSON.parse(r);
            lines2.update()
        });
        eel.get_mpc_crs(referenceFileName)().then((r) => {
            lines3.data.datasets[2].data = JSON.parse(r);
            lines3.update()

            document.getElementById("trasyxy").appendChild(document.createElement('br'));
            document.getElementById("trasycrs").appendChild(document.createElement('br'));

            let item1 = document.createElement('input');
            item1.setAttribute('id', 'mpcxy');
            item1.setAttribute('type', 'checkbox');
            item1.setAttribute('onclick', 'lines2mpc()');
            item1.setAttribute('checked', 'true');
            let item2 = document.createElement('input');
            item2.setAttribute('id', 'mpccsr');
            item2.setAttribute('type', 'checkbox');
            item2.setAttribute('onclick', 'lines3mpc()');
            item2.setAttribute('checked', 'true');
            document.getElementById("trasyxy").appendChild(item1);
            document.getElementById("trasycrs").appendChild(item2);

            let label1 = document.createElement('label');
            label1.appendChild(document.createTextNode("Model Predictive Control"));
            label1.setAttribute('for', 'mpcxy');
            label1.setAttribute('class', 'mpcxy');
            let label2 = document.createElement('label');
            label2.appendChild(document.createTextNode("Model Predictive Control"));
            label2.setAttribute('for', 'mpccsr');
            label2.setAttribute('class', 'mpccsr');
            document.getElementById("trasyxy").appendChild(label1);
            document.getElementById("trasycrs").appendChild(label2);
            document.getElementsByClassName('disable')[0].style.opacity = "1";
            document.getElementsByClassName('disable')[0].style.pointerEvents = "auto";
            document.getElementsByClassName('disable')[1].style.opacity = "1";
            document.getElementsByClassName('disable')[1].style.pointerEvents = "auto";
            document.getElementsByClassName('run')[0].style.opacity = "1";
            document.getElementsByClassName('run')[0].style.pointerEvents = "auto";
            document.getElementById('loading').style.display = "none";
        });
    }
}

function loadTrackAnalysis() {
    eel.get_laps_data(referenceFileName, tracesFileName)().then((r) => {
        trackdata = JSON.parse(r);
        document.getElementById('trasy').appendChild(makeUL(JSON.parse(r)));
        document.getElementById('loading').style.display = "none";
    });
}

function trackAdd(id) {
    let newDataset = [];
    if (document.getElementById(id).checked) {
        trackdata[Number(id)].lapData.forEach(function (item, index) {
            newDataset[index] = {
                "x": trackdata[Number(id)].lapData[index].LON,
                "y": trackdata[Number(id)].lapData[index].LAT
            }
        });
        lines1.data.datasets[Number(id) + 1].data = newDataset;
    } else {
        lines1.data.datasets[Number(id) + 1].data = [];
    }
    lines1.update();
}

function toggle(source) {
    checkboxes = document.getElementsByName('check');
    for (let i = 0, n = checkboxes.length; i < n; i++) {
        checkboxes[i].checked = source.checked;
        let newDataset = [];
        if (source.checked) {
            trackdata[Number(i)].lapData.forEach(function (item, index) {
                newDataset[index] = {
                    "x": trackdata[Number(i)].lapData[index].LON,
                    "y": trackdata[Number(i)].lapData[index].LAT
                }
            });
            lines1.data.datasets[Number(i) + 1].data = newDataset;
        } else {
            lines1.data.datasets[Number(i) + 1].data = [];
        }
        lines1.update();
    }
}

function makeUL(array) {
    document.getElementById('car').style.opacity = "1";
    document.getElementById('car').style.pointerEvents = "auto";
    document.getElementsByClassName('load')[0].style.opacity = "0";
    let list = document.createElement('div');
    let l = 0;
    for (let i = 0; i < array.length; i++) {
        let item = document.createElement('input');
        let lapSteps1 = `${i + 1}.`;
        let lapSteps2 = `${array[i]["pointsPerLap"]}`;
        let lapSteps3 = `${array[i]['averagePerpendicularDistance'].toFixed(3)}m`;
        let a = Math.floor(Math.random() * (360 - 0 + 1)) + 0;
        let b = Math.floor(Math.random() * (80 - 50 + 1)) + 50;
        let c = Math.floor(Math.random() * (60 - 40 + 1)) + 40;
        let color = 'hsl(' + a + ', ' + b + '%, ' + c + '%)';
        item.setAttribute('id', (i).toString());
        item.setAttribute('name', 'check');
        item.setAttribute('type', 'checkbox');
        item.setAttribute('onclick', 'trackAdd(this.id)');
        item.setAttribute('checked', 'true');
        list.appendChild(item);
        let label1 = document.createElement('label');
        label1.appendChild(document.createTextNode(lapSteps1));
        label1.setAttribute('class', 'label1');
        label1.setAttribute('for', (i).toString());
        label1.style.color = color;
        list.appendChild(label1);
        let label2 = document.createElement('label');
        label2.appendChild(document.createTextNode(lapSteps2));
        label2.setAttribute('class', 'label2');
        label2.setAttribute('for', (i).toString());
        label2.style.color = color;
        list.appendChild(label2);
        let label3 = document.createElement('label');
        label3.appendChild(document.createTextNode(lapSteps3));
        label3.setAttribute('class', 'label3');
        label3.setAttribute('for', (i).toString());
        label3.style.color = color;
        list.appendChild(label3);
        let br = document.createElement('br');
        list.appendChild(br);
        let dataset = [];
        trackdata[l].lapData.forEach(function (item, index) {
            dataset[index] = {
                "x": trackdata[Number(l)].lapData[index].LON,
                "y": trackdata[Number(l)].lapData[index].LAT
            }
        });
        let newDataset = {
            label: "",
            showLine: true,
            fill: false,
            "borderColor": color,
            data: dataset
        }
        l = i + 1;
        lines1.data.datasets.push(newDataset);
        lines1.update();
    }
    document.getElementById('numeroflaps').innerHTML = l.toString();
    sum = l;
    document.getElementById('track').style.opacity = "1";
    document.getElementById('track').style.pointerEvents = "auto";
    document.getElementsByClassName('load')[1].style.opacity = "0";
    return list;
}

function cp() {
    document.getElementById('parameters-text').innerHTML = "Vyberte spôsob vytvorenia modelu pomocou Simple Command Prediction:";
    selector = 1;
    if (document.getElementsByClassName('selector-buttons')[0].style.backgroundColor === 'rgb(20, 46, 59)') {
        document.getElementsByClassName('selector-buttons')[0].style.backgroundColor = '#254053';
        document.getElementsByClassName('parameters')[0].style.height = '0px';
    } else {
        document.getElementsByClassName('selector-buttons')[0].style.backgroundColor = '#142e3b';
        document.getElementsByClassName('selector-buttons')[1].style.backgroundColor = '#254053';
        document.getElementsByClassName('mpc-inputs')[0].style.height = '0px';
        document.getElementsByClassName('parameters')[0].style.height = (document.getElementsByClassName('parameters-button')[0].scrollHeight).toString() + 'px';
    }

    toggleAnimationButton('scp')
}

function mpc() {
    document.getElementById('parameters-text').innerHTML = "Vyberte spôsob vytvorenia modelu pomocou Model Predictive Control:";
    selector = 2;
    if (document.getElementsByClassName('selector-buttons')[1].style.backgroundColor === 'rgb(20, 46, 59)') {
        document.getElementsByClassName('selector-buttons')[1].style.backgroundColor = '#254053';
        document.getElementsByClassName('parameters')[0].style.height = '0px';
        document.getElementsByClassName('mpc-inputs')[0].style.height = '0px';
    } else {
        document.getElementsByClassName('selector-buttons')[0].style.backgroundColor = '#254053';
        document.getElementsByClassName('selector-buttons')[1].style.backgroundColor = '#142e3b';
        let n = document.getElementsByClassName('mpc-inputs')[0].scrollHeight + document.getElementsByClassName('parameters-button')[0].scrollHeight
        document.getElementsByClassName('mpc-inputs')[0].style.height = (document.getElementsByClassName('mpc-inputs')[0].scrollHeight).toString() + 'px';
        document.getElementsByClassName('parameters')[0].style.height = (n).toString() + 'px';
    }

    toggleAnimationButton('mpc')
}

function toggleAnimationButton(model) {
    eel.can_run_animation(model)().then((r) => {
        if (r) {
            document.getElementsByClassName('run')[0].style.opacity = "1";
            document.getElementsByClassName('run')[0].style.pointerEvents = "auto";
        } else {
            document.getElementsByClassName('run')[0].style.opacity = "0.5";
            document.getElementsByClassName('run')[0].style.pointerEvents = "none";
        }
    });
}

document.getElementById('max-button').addEventListener("click", event => {
    if (win.isMaximized())
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

document.getElementById('open-button1').addEventListener("click", event => {
    if (file1 && file2) {
        sessionStorage.setItem("reloading1", "true");
        document.location.reload();
    }
    else {
        open1();
    }
});

document.getElementById('open-button2').addEventListener("click", event => {
    if (file1 && file2) {
        sessionStorage.setItem("reloading2", "true");
        document.location.reload();
    }
    else {
        open2();
    }
});

window.onload = function() {
    var reloading1 = sessionStorage.getItem("reloading1");
    if (reloading1) {
        sessionStorage.removeItem("reloading1");
        open1();
    }
    var reloading2 = sessionStorage.getItem("reloading2");
    if (reloading2) {
        sessionStorage.removeItem("reloading2");
        open2();
    }
}

function open1() {
    dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [
            {name: 'Log File (.log)', extensions: ['log']},
        ]
    }).then(result => {
        if (!result.canceled) {
            console.log('check')
            referenceFileName = result.filePaths.toString()

            // Path for Linux/Mac or Windows
            if (referenceFileName.split(/(.*)\\/)[2] == undefined) {
                document.getElementById('file-name1').innerHTML = referenceFileName.split(/(.*)\//)[2].split(/\.log$/)[0];
            } else {
                document.getElementById('file-name1').innerHTML = referenceFileName.split(/(.*)\\/)[2].split(/\.log$/)[0];
            }
            eel.get_reference_laps(referenceFileName)().then((r) => {
                document.getElementById('referencenumber').innerHTML = r;
            });
            file1 = 1;
            if (file1 && file2) {
                document.getElementsByClassName('disabled')[0].style.opacity = "1";
                document.getElementsByClassName('disabled')[1].style.opacity = "1";
                document.getElementsByClassName('disabled')[1].style.pointerEvents = "auto";
                document.getElementsByClassName('disabled')[2].style.opacity = "1";
                document.getElementsByClassName('disabled')[2].style.pointerEvents = "auto";
            }
            (referenceFileName, 0);
            loadReference(referenceFileName);
        }
    }).catch(err => {
        console.log(err)
    })
}

function open2() {
    dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [
            {name: 'Log File (.log)', extensions: ['log']},
        ]
    }).then(result => {
        if (!result.canceled) {
            tracesFileName = result.filePaths.toString()
            // Path for Linux/Mac or Windows
            if (tracesFileName.split(/(.*)\\/)[2] == undefined) {
                document.getElementById('file-name2').innerHTML = tracesFileName.split(/(.*)\//)[2].split(/\.log$/)[0];
            } else {
                document.getElementById('file-name2').innerHTML = tracesFileName.split(/(.*)\\/)[2].split(/\.log$/)[0];
            }
            file2 = 1;
            if (file1 && file2) {
                document.getElementsByClassName('disabled')[0].style.opacity = "1";
                document.getElementsByClassName('disabled')[1].style.opacity = "1";
                document.getElementsByClassName('disabled')[1].style.pointerEvents = "auto";
                document.getElementsByClassName('disabled')[2].style.opacity = "1";
                document.getElementsByClassName('disabled')[2].style.pointerEvents = "auto";
            }
            loadChartForFile(referenceFileName, 0);
        }
    }).catch(err => {
        console.log(err)
    })
}

document.getElementById('back-button').addEventListener("click", event => {
    if (document.getElementById('ww2').style.display === "block")
        vozidlodesc = document.getElementById('description').value
    if (document.getElementById('ww1').style.display === "block")
        drahadesc = document.getElementById('description').value
    document.getElementById('w1').style.display = "block";
    document.getElementById('w2').style.display = "none";
});

document.getElementById('track').addEventListener("click", event => {
    document.getElementById('name').innerHTML = "Analýza dráhy";
    document.getElementById('w1').style.display = "none";
    document.getElementById('w2').style.display = "block";
    document.getElementById('ww1').style.display = "block";
    document.getElementById('ww2').style.display = "none";
    document.getElementById('description').value = drahadesc;
    if (referenceFileName !== undefined && car === 0) {
        document.getElementById('loading').style.display = "flex";
        loadTrackAnalysis()
        car = 1;
    }
});

document.getElementById('car').addEventListener("click", event => {
    document.getElementById('name').innerHTML = "Analýza vozidla";
    document.getElementById('w1').style.display = "none";
    document.getElementById('w2').style.display = "block";
    document.getElementById('ww1').style.display = "none";
    document.getElementById('ww2').style.display = "block";
    document.getElementById('description').value = vozidlodesc;
});

document.getElementById('print-button').addEventListener("click", event => {
    if (window.getComputedStyle(document.getElementById('ww2')).display === "block") {
        lines2.canvas.parentNode.style.width = '185mm';
        lines3.canvas.parentNode.style.width = '185mm';
        Chart.instances[1].resize();
        Chart.instances[2].resize();
        window.print();
        lines2.canvas.parentNode.style.width = '100%';
        lines3.canvas.parentNode.style.width = '100%';
        Chart.instances[1].resize();
        Chart.instances[2].resize();
    } else {
        lines1.canvas.parentNode.style.width = '185mm';
        Chart.instances[0].resize();
        window.print();
        lines1.canvas.parentNode.style.width = '100%';
        Chart.instances[0].resize();
    }
});

document.getElementById('save-button').addEventListener("click", event => {
    if (document.getElementById('ww1').style.display === "block") {
        let selected = lines1.data.datasets.reduce(function (arr, e, i) {
            if (e.data.length > 0 && i > 0) arr.push(i - 1);
            return arr;
        }, []);
        dialog.showOpenDialog({
            title: "Select a folder",
            properties: ["openDirectory"]
        }).then(result => {
            if (!result.canceled) {
                eel.export_data(
                    result.filePaths.toString(),
                    referenceFileName.split(/(.*)\\/)[2].split(/\.log$/)[0],
                    document.getElementById('description').value,
                    selected)
            }
        });
    }
    if (document.getElementById('ww2').style.display === "block") {
        dialog.showOpenDialog({
            title: "Select a folder",
            properties: ["openDirectory"]
        }).then(result => {
            if (!result.canceled) {
                eel.export_mpc(result.filePaths.toString(), referenceFileName.split(/(.*)\\/)[2].split(/\.log$/)[0])
                eel.export_predicted_data(
                    result.filePaths.toString(),
                    document.getElementById('description').value)
            }
        });
    }
})

function animate_car_movement() {
    eel.get_track_coordinates(referenceFileName)
    if (selector === 1) {
        model = 'scp'
        eel.animate_track(model)
    }
    if (selector === 2) {
        model = 'mpc'
        eel.animate_track(model)
    }
}