import sys
sys.path.append(".")

import cProfile as prof
import pandas as pd
import analysis.log_file_analyzer as l
import analysis.lap_difference_analyzer as l2
import old_functions as old
import timeit
import numpy as np
import math
from numpy import sqrt as nsqrt
from numpy import arctan2
from math import atan2
from math import sqrt as msqrt
from old_functions import find_out_difference_perpendiculars

ref = pd.read_csv('benchmark/ref1.csv')
data = l.get_essential_data('benchmark/200629130554_gps.log')

data = data[36700:]
data.reset_index(drop=True, inplace=True)


def run_with_profiler():
    prof.run("l2.find_out_difference_perpendiculars(data, ref)")
    prof.run("old.find_out_difference_perpendiculars(data, ref)")


# run_with_profiler()


############# Angle between vectors #######################
def calculate_angles_v1():
    l2.find_angle_between_vectors([1, 0], [1, 0])
    l2.find_angle_between_vectors([0.5, 0.1], [1, 1.1])


def calculate_angles_v2():
    old.find_angle_between_vectors((1, 0), (1, 0))
    old.find_angle_between_vectors((0.5, 0.1), (1, 1.1))


def angle_v3(v1, v2):
    return np.math.atan2(np.linalg.det([v1,v2]), np.dot(v1,v2))


def calculate_angles_v3():
    angle_v3((1, 0), (1, 0))
    angle_v3((0.5, 0.1), (1, 1.1))


def benchmark_angle_calculation():
    print("angle test:")
    result = timeit.timeit(lambda: calculate_angles_v1(), number=10000)
    print("calculate_angles_v1 {}s".format(result))
    result = timeit.timeit(lambda: calculate_angles_v2(), number=10000)
    print("calculate_angles_v2 {}s".format(result))
    result = timeit.timeit(lambda: calculate_angles_v3(), number=10000)
    print("calculate_angles_v3 {}s".format(result))
    print("#####################")

#######################################################

############# SQRT #######################
def benchmark_sqrt_calculation():
    print("sqrt test:")
    result = timeit.timeit(lambda: nsqrt(42.345), number=10000)
    print("nsqrt {}s".format(result))
    result = timeit.timeit(lambda: msqrt(42.345), number=10000)
    print("msqrt {}s".format(result))
    result = timeit.timeit(lambda: math.sqrt(42.345), number=10000)
    print("math.sqrt {}s".format(result))
    print("#####################")

##########################################

############# atan2 #######################
def benchmark_atan2_calculation():
    print("atan2 test:")
    result = timeit.timeit(lambda: math.atan2(42.345, 23.2), number=10000)
    print("math.atan2 {}s".format(result))
    result = timeit.timeit(lambda: atan2(42.345, 23.2), number=10000)
    print("atan2 {}s".format(result))
    result = timeit.timeit(lambda: arctan2(42.345, 23.2), number=10000)
    print("arctan2 {}s".format(result))
    print("#####################")

##########################################

############# indexing #######################
def benchmark_index_calculation():
    print("Indexing test:")
    d = { "a":range(5000), "b": range(5000)}
    df = pd.DataFrame(d)
    l = list(range(5000))

    result = timeit.timeit(lambda: df.loc[2345], number=10000)
    print("df.loc {}s".format(result))
    result = timeit.timeit(lambda: df.iloc[2345], number=10000)
    print("df.iloc {}s".format(result))
    result = timeit.timeit(lambda: l[2345], number=10000)
    print("list index {}s".format(result))
    print("#####################")

##########################################

# benchmark_angle_calculation()
# benchmark_sqrt_calculation()
# benchmark_atan2_calculation()
# benchmark_index_calculation()