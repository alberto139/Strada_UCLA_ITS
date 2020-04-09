import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import csv, time
#import altair as alt
import pandas as pd
#import fire_push

a = (331, 211)
b = (388, 312)
c1 = (342, 266)
c2 = (393, 257)

def is_left(line, c):
    #print(line)
    #print(c)
    a, b = line
    left = ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0
    return left

# TODO: Fix crossing two lines BUG
def class_count(aois, detections, df, upload):

    for aoi in aois:
        for obj in detections:
            if len(obj.prev_centers) > 2 and (not obj.id in aoi.objects_crossed):
                current_center = obj.prev_centers[-1]
                prev_center = obj.prev_centers[-2]

                point = Point(prev_center[0], prev_center[1])
                #print("aoi.label: " + str(aoi.label))
                #print("aoi.area: " + str(aoi.area))
                polygon = Polygon(aoi.area)

                is_prev_left = is_left(aoi.line, prev_center)
                is_curr_right = not is_left(aoi.line, current_center)

                if is_prev_left and is_curr_right and polygon.contains(point) \
                 and (not obj.id in aoi.objects_crossed) and (obj.cls in aoi.relevant_classes):
                    direction = 'right'
                    update_obj_crossing(obj, aoi, direction)
                    crossing_info(obj, aoi, polygon, point, prev_center, current_center)
                    #record_df(df, obj, aoi, prev_center, current_center)

                    #if upload:
                    #    fire_push.push_data(df, obj, aoi, prev_center, current_center)
                
                elif not is_prev_left and not is_curr_right and polygon.contains(point) \
                 and (not obj.id in aoi.objects_crossed) and (obj.cls in aoi.relevant_classes):
                    direction = 'left'
                    update_obj_crossing(obj, aoi, direction)
                    crossing_info(obj, aoi, polygon, point, prev_center, current_center)
                    #record_df(df, obj, aoi, prev_center, current_center)

                    #if upload:
                    #    fire_push.push_data(df, obj, aoi, prev_center, current_center)
    return aois

# Count objects
# Keep track of when objects enter and area and when they leave it
# params: 
#           aois - objects of class containing area infromation
#           detections - objects of object class containing vehicle infromation
#           df - local dataframe to store counts
#           upload - flag to upload to firebase
# returns:
#           None
def class_count_area(aois, detections, df, upload, frame_time):

    # iterate over all areas and all objects
    # if an object is seen in an area for the first time, keep track of it
    # if an object previously seen in an area is no longer in the area, record its data in dataframe
    for aoi in aois:
        aoi.active = False
        for obj in detections:
            #obj.active = False
            if len(obj.prev_centers) > 2:
                current_center= obj.prev_centers[-1]
                prev_center = obj.prev_centers[-2]

                point = Point(prev_center[0], prev_center[1])

                # special case for bike lane area
                # since objects are in the area likely to corss the bike lane in order to get to the parking spaces
                # only tracked cars parked in bile lane (assume they are in the bike lane for more than 20 seconds)
                if aoi.label == 'bike_lane':
                    None

                if not aoi.label == 'space_1' and not aoi.label == 'space_2':
                    continue


                    
                # First time seeing object in area
                if aoi.polygon.contains(point) and (not obj.id in aoi.objects_crossed):
                    aoi.objects_crossed.append(obj.id)
                    obj.in_time = frame_time
                    obj.active = True
                    print("Object " + str(obj.id) + " entered area " + str(aoi.label))
                
                # seeing object in are NOT for the first time
                elif aoi.polygon.contains(point):
                    obj.active = True
                    aoi.active = True




def update_obj_crossing(obj, aoi, direction):
    obj.direction = direction
    obj.directions.append(direction)
    obj.crossed = True
    obj.cross_time = time.time()
    obj.str_time = pd.to_datetime(time.time() - (3600 * 7), unit='s')
    aoi.crossed = True
    aoi.objects_crossed.append(obj.id)
    obj.aoi_ids.append(aoi.label)
    obj.cross_times.append(time.time())
    


def crossing_info(obj, aoi, polygon, point, prev_center, current_center):
    print('-----------crossed ------------')
    print('Aoi label: ' + str(obj.aoi_ids))
    print('object direction: ' + str(obj.direction))
    print('object id: ' + str(obj.id))
    print('object cls: ' + str(obj.cls))
    print('relevant objs: ' + str(aoi.relevant_classes))
    print('line number: ' + str(aoi.id))
    #print('path: ' + str(obj.prev_centers))
    print('prev_center: ' + str(prev_center))
    print('current_center: ' + str(current_center))
    print('line: ' + str(aoi.line))
    print('is_prev_left: ' + str(is_left(aoi.line, prev_center)))
    print('is_curr_right: ' + str(not is_left(aoi.line, current_center)))
    print('in poly:' + str(polygon.contains(point)))
    print('crossed this line: ' + str(obj.id in aoi.objects_crossed))
    print('objs corssed: ' + str(aoi.objects_crossed))
    #obj.crossed = True
    
    print(obj.cross_time)
    
    print('objs corssed: ' + str(aoi.objects_crossed))

    cv2.imshow("subimg", obj.subimg)
    cv2.waitKey(1)

# TODO: for larger loops fill a temp dataframe using iloc, then append to permanent dataframe and clear temp every 1000 iterations
def record_df(df, obj, aoi, prev_center, current_center):
    #col_names =  ['timestamp', 'id', 'class', 'aoi_id', 'speed']
    df.loc[len(df)] = [obj.str_time, obj.id, cls2label(obj.cls), aoi.id, obj.speed, obj.direction]

    #serve_charts(df)
    df.to_csv('test.csv', index = False)

def serve_charts(df):
    chart = alt.Chart(df).mark_point().encode(
        x='timestamp'
    ).interactive()

    chart.serve()



def cls2label(cls_num):
    cls_num = int(cls_num)
    if cls_num == 1:
        return 'person'
    elif cls_num == 2:
        return 'car'
    elif cls_num == 3:
        return 'cyclist'
    elif cls_num == 4:
        return 'bus'
    else:
        return 'None'
