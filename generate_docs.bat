mkdir documentation
python -m pydoc -w main ^
    animation ^
    animation.animation_rendering ^
    mpc ^
    mpc.cubic_spline_planner ^
    mpc.mpc
move *.html documentation

