import numpy as np 
import cv2
from collections import deque

def difference(img1, img2):
    """
    Calculates the difference between two images.
    img1: numpy array
    img2: numpy array
    Returns: new image displaying the differences (high pixel values = very different)
    """
    viz = False

    # Grayscale the two images
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype('float32')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype('float32')
    
    # Normalize
    gray1 /= 255
    gray2 /= 255


    # Get the difference in magnitude for every pixel. int16 so that negative don't wrap around
    difference = abs(np.subtract(gray1, gray2))

    result = difference
    # Make all the pixels under a threshold zero
    # NOTE: This works but makes everthing really slow

    #result = np.array([[0 if pixel < 0.2 else pixel for pixel in column] for column in difference])
    result[result < 0.1] = 0 # Same thing as the line above but WAY faster. I <3 numpy
    

    if viz:  
        cv2.imshow('difference', result)#.astype(np.int8))
        cv2.waitKey(0)

    score = sum(sum(result)) # Sum all the pixels together. Similar frames will have a low score

    return result, score


def get_average_frame(past_frames):
    """
    Updates the average frame from the past N frames.
    If there is no average frame it creates one.
    past_frames: queue storing the past frames.
    frames: current frame to be added to the average
    Returns: ret (boolean if there is an average), average (the new average frame)
    """
    viz = False
    average_frame = None
    
    # Get the average of the previous N frames
    # print(np.amax(sum(past_frames)))
    N = len(past_frames)
    average_frame = sum(past_frames) /(255 * N)

    if viz:
        cv2.imshow('average_frame', average_frame.astype('float32'))
        cv2.waitKey(0)

    return average_frame



def update_past_frames(past_frames, frame, length):

    if len(past_frames) < length:
        past_frames.append(frame.astype('float32'))
        
    else:
        past_frames.popleft()
        past_frames.append(frame.astype('float32'))

    return past_frames


def motion_based_detection(img, count, past_frames):

    if count > 50 and count < 100:
        img = gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')
        past_frames = update_past_frames(past_frames, img, 50)
        average = get_average_frame(past_frames) * 255

        cv2.imshow("average", average)
        cv2.waitKey(0)

    # Get the background from taking the average

    # Apply gaussian blur to make blobs less sparce
    #img = cv2.GaussianBlur(img, (25, 5), 0)

    #filter mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Fill any small holes
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    #opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    #dilation = cv2.dilate(opening, kernel, iterations=2)

    # threshold
    #th = dilation[dilation < 240] = 0

    return img
    #return blurred_frame



def test():

    #img1 = cv2.imread('/home/alberto/Desktop/data/frames/2018-08-11 20:29:11.png')
    img1 = cv2.imread('/home/alberto/Desktop/data/frames/20.png')
    img2 = cv2.imread('/home/alberto/Desktop/data/frames/30.png')
    img3 = cv2.imread('/home/alberto/Desktop/data/frames/40.png')
    img4 = cv2.imread('/home/alberto/Desktop/data/frames/50.png')
    img5 = cv2.imread('/home/alberto/Desktop/data/frames/60.png')
    img6 = cv2.imread('/home/alberto/Desktop/data/frames/70.png')
    img7 = cv2.imread('/home/alberto/Desktop/data/frames/300.png')
    img8 = cv2.imread('/home/alberto/Desktop/data/frames/301.png')
    img9 = cv2.imread('/home/alberto/Desktop/data/frames/302.png')
    img10 = cv2.imread('/home/alberto/Desktop/data/frames/312.png')


    frames = deque()
    length = 5
    update_past_frames(frames, img7, length)
    update_past_frames(frames, img8, length)

    new_frame = img10
    average = get_average_frame(frames) * 255
    
    result, score = difference(average, new_frame)
    print('Difference Score: ' + str(score))


#test()