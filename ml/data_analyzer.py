import pandas as pd
import math 
import numpy as np
from numpy.linalg import norm
from scipy import stats
from obspy.geodetics import degrees2kilometers
import matplotlib.pyplot as plt
from similaritymeasures import curve_length_measure, frechet_dist
from similaritymeasures import area_between_two_curves
from os import listdir, remove
from shutil import copy
from time import time_ns
import threading as thread

def init_dataframe_old(path):
    df = pd.read_csv(path, header=None, sep=';')
    df = df.drop(columns=[0, 1, 3, 5, 7, 9, 11, 13, 15, 17])
    df.columns = ['LAT', 'LON', 'UTMX', 'UTMY', 'HMSL',
                  'GSPEED', 'CRS', 'HACC', 'NXPT']
    return df

def init_dataframe(path):
    df = pd.read_csv(path, sep=',')
    return df

def create_and_save_images():
    for fileName in listdir('data/laps'):
        df = init_dataframe('data/laps/{}'.format(fileName))
        df.plot(x='LAT', y='LON')
        plt.savefig('data/laps_images/{}'.format(fileName[:-4]))

# df0 = init_dataframe_old('data/feri_logy_analyza/200629_karting/200629130554_gps.log')
# df1 = init_dataframe_old('data/feri_logy_analyza/200629_karting/kartfinal36pr3r.csv')
# df2 = init_dataframe_old('data/200623_academy/200623110845_gps.log')
# df3 = init_dataframe_old('data/200623_academy/200623111152_gps.log')
# df4 = init_dataframe_old('data/200623_academy/200623120513_gps.log')
# df5 = init_dataframe_old('data/200623_academy/200623121213_gps.log')
# df6 = init_dataframe_old('data/200623_academy/200623122206_gps.log')
# df7 = init_dataframe_old('data/200623_academy/trasa23.csv')

# df0.plot(x='LAT', y='LON')
# df1.plot(x='LAT', y='LON')
# df2.plot(x='LAT', y='LON')
# df3.plot(x='LAT', y='LON')
# df4.plot(x='LAT', y='LON')
# df5.plot(x='LAT', y='LON')
# df6.plot(x='LAT', y='LON')
# df7.plot(x='LAT', y='LON')

ref1 = init_dataframe('data/ref1.csv')
ref2 = init_dataframe('data/ref2.csv')
# traces1 = init_dataframe('data/traces1.csv')
# traces2_1 = init_dataframe('data/traces2-1.csv')
# traces2_2 = init_dataframe('data/traces2-2.csv')

# traces2_1.plot(x='LAT', y='LON')
# traces2_2.plot(x='LAT', y='LON')
# plt.show()

# fig = plt.figure()
# for frame in [df2, df7]:
#     plt.plot(frame['LON'], frame['LAT'])
#plt.show()

def create_images_with_refference_lap():
    for fileName in listdir('data/laps'):
        df = init_dataframe('data/laps/{}'.format(fileName))
        ref = ref1 if fileName.startswith('lap1') else ref2
        _ = plt.figure()
        for frame in [df, ref]:
            plt.plot(frame['LON'], frame['LAT'])
        plt.savefig('data/laps_images_with_ref_lap/{}'.format(fileName[:-4]))
        plt.close()
# create_images_with_refference_lap()

def create_curve(dataframe):
    curve = np.zeros((dataframe.shape[0], 2))
    curve[:, 0] = dataframe.LON
    curve[:, 1] = dataframe.LAT
    return curve

def find_out_difference():
    ref1 = init_dataframe('data/ref1.csv')
    ref2 = init_dataframe('data/ref2.csv')

    ideal_curve_ref1 = create_curve(ref1)
    ideal_curve_ref2 = create_curve(ref2)
    laps_dir = 'data/laps'

    name_column = 'Name'
    measurement_column = 'Measurements count'
    frechet_column = 'Frechet distance'
    curve_len_column = 'Curve length measure'
    area_column = 'Area diff'
    data_structure = {name_column: [], measurement_column:[],\
                     frechet_column: [], curve_len_column: [],\
                     area_column: []}

    differences_df = pd.DataFrame(data=data_structure)

    for file_name in listdir(laps_dir):
        lap = init_dataframe('{}/{}'.format(laps_dir, file_name))
        experimental_curve = create_curve(lap)

        ideal_curve = ideal_curve_ref1 if file_name.startswith('lap1') \
                                       else ideal_curve_ref2
        m_count = len(lap)
        fd = frechet_dist(experimental_curve, ideal_curve)
        cl = curve_length_measure(experimental_curve, ideal_curve)
        area = area_between_two_curves(experimental_curve, ideal_curve) 

        difference = {name_column: file_name, measurement_column: m_count,\
                    frechet_column: fd, curve_len_column: cl,\
                    area_column : area}

        differences_df = differences_df.append(difference, ignore_index=True)
        print(file_name)

    differences_df.to_csv('data/differences.csv', index=False)

# create_and_save_images()
# find_out_difference()

def init_dataframe_differences():
    df = pd.read_csv('data/differences.csv')
    return df

# differences_df = init_dataframe_differences()
# differences_df.plot.scatter(x='Frechet distance', y='Curve length measure')
# plt.show()

# fig = plt.figure()
# for frame in [df, df1]:
#     plt.plot(frame['LON'], frame['LAT'])

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(df['LON'], df['LAT'], df['GSPEED'])

# df1['NXPT'].plot()

# plt.show()

############################################################################################################################

def line_length(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_closest_point(point, lap):
    minIndex = 0
    minLength = math.inf
    for i in lap.index:
        lat = lap.loc[i].LAT
        lon = lap.loc[i].LON

        length = line_length(lat, lon, point.LAT, point.LON)
        if length < minLength:
            minIndex = i
            minLength = length

    return minIndex

def find_angle_between_vectors(vector_A, vector_B):
    unit_vector_A = vector_A / np.linalg.norm(vector_A) 
    unit_vector_B = vector_B / np.linalg.norm(vector_B)
    dot_product = np.dot(unit_vector_A, unit_vector_B)
    return np.arccos(dot_product)                       #return angle in radians 

def create_vector(point_A, point_B):
    return [point_B.LAT - point_A.LAT, point_B.LON - point_A.LON]

# Perdendicular from p1 to line (p2,p3)
def shortest_distance(p1,p2,p3):  
    dist = norm(np.cross(p2-p3, p3-p1))/norm(p3-p2)
    return dist

def find_shortest_distance(p1, p2, p3):
    x = np.array([p1.LAT, p1.LON])
    y = np.array([p2.LAT, p2.LON])
    z = np.array([p3.LAT, p3.LON])
    return shortest_distance(x,y,z)

def lap1_generator():
    for file_name in listdir('data/laps'):
        if not file_name.startswith('lap1'): continue
        yield (file_name, init_dataframe('data/laps/'+file_name))

def lap2_generator():
    for file_name in listdir('data/laps'):
        if file_name.startswith('lap1'): continue
        yield (file_name, init_dataframe('data/laps/'+file_name))
        
def find_out_difference_v2(lap, ref_lap):
    sum_of_distances = 0
    distances = []

    for i in lap.index:
        point = lap.loc[i]

        closest_index = find_closest_point(point, ref_lap)
        closest_point = ref_lap.loc[closest_index]

        neighbor_i = len(ref_lap)-1 if closest_index == 0 else closest_index-1
        neighbor1 = ref_lap.loc[neighbor_i]
        neighbor_i = 0 if len(ref_lap) == closest_index+1 else closest_index+1
        neighbor2 = ref_lap.loc[neighbor_i]

        v1 = create_vector(closest_point, point)
        v2 = create_vector(closest_point, neighbor1)
        v3 = create_vector(closest_point, neighbor2)

        angle1 = find_angle_between_vectors(v1,v2)
        angle2 = find_angle_between_vectors(v1,v3)

        degrees90 = math.pi / 2
        min_dist = -1
        if angle1 > degrees90 and angle2 > degrees90:
            min_dist = line_length(point.LAT, point.LON, closest_point.LAT, closest_point.LON)
        elif angle1 < degrees90 and angle2 < degrees90:
            dist1 = find_shortest_distance(point, closest_point, neighbor1)
            dist2 = find_shortest_distance(point, closest_point, neighbor2)
            min_dist = dist1 if dist1 <= dist2 else dist2
        elif angle1 <= degrees90:
            min_dist = find_shortest_distance(point, closest_point, neighbor1)
        elif angle2 <= degrees90:
            min_dist = find_shortest_distance(point, closest_point, neighbor2)

        if min_dist == -1:
            print ('ERROR: Could not find distance')       
            print("Indices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        elif math.isnan(min_dist):
            print("NAN value!!!\nIndices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        elif min_dist < 0:
            print("Negative value!!!\nIndices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        else:
            min_dist = degrees2kilometers(min_dist) * 100000    # in centimeters
            sum_of_distances += min_dist   
            distances.append(min_dist)     
    
    print ("Sum is {}cm".format(sum_of_distances))
    print ("Average distance from reference lap is {}cm".format(sum_of_distances/len(lap)))
    return distances

differences_dict = {}
max_len = 2500

find_dff = lambda x,y,z: z.extend(find_out_difference_v2(x,y))

for file_name, lap in lap2_generator():
    diff1 = []
    diff2 = []

    th1 = thread.Thread(target=find_dff, args=(lap, ref2, diff1))
    th2 = thread.Thread(target=find_dff, args=(ref2, lap, diff2))

    th1.start()
    th2.start()
    th1.join() 
    th2.join() 

    m = map(lambda x: str(), range(max_len - len(diff2)))
    diff2 += list(m)
    
    m = map(lambda x: str(), range(max_len - len(diff1)))
    diff1 += list(m)

    differences_dict[file_name] = diff1
    differences_dict[file_name+' (from view of ref lap)'] = diff2

diff_df = pd.DataFrame(differences_dict)
diff_df.to_csv('data/detail_diff2.csv', index=False)

############################################################################################################################
