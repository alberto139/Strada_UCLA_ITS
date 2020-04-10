import video_functions, utils, object_class, time, aoi, counting
from collections import deque
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
#import fire_push

def single():

    upload = False
    model_path = "frozen_inference_graph.pb" #mobilnet_fpn w/ traffic cameras
    cap = cv2.VideoCapture('/home/alberto/Desktop/ITS/its_video.AVI')

    # ITS video start time
    its_time_human = '30.08.2019 17:00:04'
    its_time_epoch = utils.human_to_epoch(its_time_human)


    ### Camera Setup ###
    width = 1920 // 2
    height = 1080 // 2
    cap.set(3,width)
    cap.set(4,height)
    #cap.set(cv2.CAP_PROP_FPS, 10)

    ####################
    #img = cv2.imread('castro3.png')
    img = cv2.imread('its.png')
    #img = utils.rotate(img)
    ### JSON Files ###
    #line_json = 'jsons/cross_lines.json'
    area_json = 'jsons/its_areas_new.json'

    print("image size")
    print((width, height))
    
    out = cv2.VideoWriter('demo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640, 360))
    ##################
    past_frames = deque()
    first_detection = True
    motion = False
    motion_based_detection = False
    remember_n_frames = 5
    prev_detections = []
    conf_threshold = 0.5
    max_id = 0
    total_count = 0
    mul = 2
    score = 1

    img, lines, areas, aois = aoi.aoi(img, mul, area_json, [])
    draw_boxes = True
    draw_areas = True
    draw_lines = False
    draw_text = True
    draw_paths = False

    # Pandas dataframe setup
    col_names =  ['timestamp', 'id', 'class', 'aoi_id', 'speed', 'direction']
    df = pd.DataFrame(columns = col_names)
    
    
    # setup
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections','detection_boxes','detection_scores','detection_classes']:  # Note: might need to ad detection_masks
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            start_time = time.time()
            
            while True:
                # capture
                total_count += 1
                # load full image
                ret, img = cap.read()
                #img = utils.rotate(img)
                # resize image
                img = cv2.resize(img, (img.shape[1] // mul, img.shape[0] // mul))
                # load resize lines and draw
                #cv2.imshow('capture', img)
                #cv2.waitKey(1)

                # detection
                if score > 0:
                    output_dict = sess.run(
                    tensor_dict, feed_dict={
                    image_tensor: np.expand_dims(img, 0)})
                    output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]

                    # tracking
                    new_detections = []
                    for i, element in enumerate(output_dict['detection_boxes']):
                        if output_dict['detection_scores'][i] > conf_threshold:
                            temp_object = object_class.Object(img, output_dict['detection_boxes'][i], output_dict['detection_classes'][i])
                            # if not a car, skip
                            #if not temp_object.cls_string == 'car':
                            #    continue
                            new_detections.append(temp_object)

                    if first_detection:
                        for thing in new_detections:
                            thing.id = max_id
                            max_id += 1
                            prev_detections.append(thing)
                            first_detection = False

                    prev_detections, max_id = utils.match(prev_detections, new_detections, max_id, time.time())
                    #prev_detections = []

                    # nms
                    boxes = []
                    for obj in prev_detections:
                        boxes.append(obj.box)
                    pick = utils.non_max_suppression_fast(np.array(boxes))
                    prev_detections = [prev_detections[x] for x in pick]

                    #speed estimation
                    #prev_detections = utils.speed_estimation(prev_detections)

                    # counting
                    #counting.count(prev_detections, lines, areas)
                    #counting.class_count(aois, prev_detections, df, upload)
                    counting.class_count_area(aois, prev_detections, df, upload, its_time_epoch)

                # increase its start time every frame
                its_time_epoch += 1

                # visualization
                key_press = cv2.waitKey(1)
                if key_press == 98: # b
                    draw_boxes = not draw_boxes
                if key_press == 108: # l
                    draw_lines = not draw_lines
                if key_press == 97: # a
                    draw_areas = not draw_areas
                if key_press == 116: # t
                    draw_text = not draw_text
                if key_press == 112: # p
                    draw_paths = not draw_paths
                

                
                
                if draw_areas:
                    img = utils.draw_aoi_active(img, aois)
                if draw_text:
                    img = utils.draw_text_active(img, prev_detections)
                    #img = utils.draw_text(img, prev_detections)
                if draw_paths:
                    img = utils.draw_paths_active(img, prev_detections)
                    #img = utils.draw_paths(img, prev_detections)
                if draw_boxes:
                    img = utils.draw_boxes_active(img, prev_detections)
                    #img = utils.draw_boxes(img, prev_detections)

                if key_press == 115: # s
                    cv2.imwrite(str(time.time()) + "screen_shot.png", img)


                cv2.imshow('detection', img)
                cv2.waitKey(1)
                #print("FPS: " + str( total_count /  (time.time() -  start_time)))
                #print("Time Elapsed: " + str(time.time() - start_time))
                #print(img.shape)
                out.write(img)

single()