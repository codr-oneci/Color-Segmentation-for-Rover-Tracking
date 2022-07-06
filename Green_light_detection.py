# Python program to explain cv2.imshow() method 
  
# importing cv2 
import cv2 
import numpy as np
import sep
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Reading an image in default mode
frame = cv2.imread("rover_green_light_3_Color.png")


hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
cv2.imshow("Image Window", frame) 
 
# Threshold of blue in HSV space
lower_blue = np.array([70, 0, 255])
upper_blue = np.array([160, 255, 256])
 
# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(frame, frame, mask = mask)
gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
objects = sep.extract(gray_result, 1.5, 150) 
 
cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow('result', result)

# plot background-subtracted image
fig, ax = plt.subplots()
m, s = np.mean(gray_result), np.std(gray_result)
im = ax.imshow(gray_result, interpolation='nearest', cmap='gray',
               vmin=m-s, vmax=m+s, origin='lower')

# plot an ellipse for each object
for i in range(len(objects)):
    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                width=6*objects['a'][i],
                height=6*objects['b'][i],
                angle=objects['theta'][i] * 180. / np.pi)
    e.set_facecolor('none')
    e.set_edgecolor('red')
    ax.add_artist(e)
plt.show()


#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 
