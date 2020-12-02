import sys
sys.path.append(".")

import cProfile as prof
from timeit import timeit as timer
from sympy import Point, Segment
import pandas as pd
import numpy as np
import analysis.log_file_analyzer as l
import analysis.lap_difference_analyzer as l2
import old_functions as old


ref = pd.read_csv('benchmark/ref1.csv')
data = l.get_essential_data('benchmark/200629130554_gps.log')

data = data[33060:]
data.reset_index(drop=True, inplace=True)

def run_with_profiler():
    prof.run("old.separate_laps(data, ref)")  
    prof.run("l.separate_laps(data, ref)")  


# run_with_profiler()

#################### INTERSECTION ##############################
point_top = Point(np.array([0, 4]), evaluate=False)
point_bottom = Point(np.array([-1, 1]), evaluate=False)
start_line = Segment(point_top, point_bottom, evaluate=False)


def intersection_v1():
    point1 = Point(-2, 3, evaluate=False)
    point2 = Point(5, 3, evaluate=False)
    segment = Segment(point1, point2, evaluate=False)
    segment.intersection(start_line)


def intersection_v2():
    s1 = l.segment((-2, 3), (5, 3))
    s2 = l.segment([0, 4], [-1, 1])
    l.intersection(s1, s2)
    

def benchmark_intersection_calculation():
    result = timer(lambda : intersection_v1(), number=2000)
    print("intersection_v1 {}s".format(result))
    result = timer(lambda : intersection_v2(), number=2000)
    print("intersection_v2 {}s".format(result))

################################################################

# benchmark_intersection_calculation()