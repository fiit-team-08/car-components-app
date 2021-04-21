function animate_car_movement() {
    if (selector === 1) {
        model = 'scp'
        eel.animate_track(model)
    }
    if (selector === 2) {
        model = 'mpc'
        eel.animate_track(model)
    }
}


//window.addEventListener("keydown", function(event) {
//    if (event.key == 'x') {
//        console.log(`KeyboardEvent: key='${event.key}' | code='${event.code}'`)
//    }
//}, true);