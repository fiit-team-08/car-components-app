new Chart(document.getElementById("chart"), {
    "type":"line","data": {
        "labels":["1","2","3","4","5","6","7"],"datasets":[ {
            "label":"My First Dataset","data":[65,59,80,81,56,55,40],"fill":false,"borderColor":"#254053e6","lineTension":0.1}

        ]}

    ,"options": {
        legend: {
            display: false
        }
    }
});