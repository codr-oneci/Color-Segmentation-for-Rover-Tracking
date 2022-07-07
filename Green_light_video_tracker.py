# Python program to explain cv2.imshow() method 

# importing cv2 
import cv2 
import numpy as np
import sep
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pyrealsense2 as rs



# Create an object to read 
# from camera

# objects for storing videos in memory
result_ellipse = cv2.VideoWriter('sources_ellipsoids.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640,480))


result_depth = cv2.VideoWriter('depth.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640,480))

result_color = cv2.VideoWriter('color.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640,480))

result_mask = cv2.VideoWriter('mask.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640,480))

filter_result = cv2.VideoWriter('result_color_segmentation.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640,480))

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


try:
	while True:

	 	# Wait for a coherent pair of frames: depth and color
        	frames = pipeline.wait_for_frames()
        	depth_frame = frames.get_depth_frame()
        	color_frame = frames.get_color_frame()
        	if not depth_frame or not color_frame:
            		continue

        	# Convert images to numpy arrays
        	depth_image = np.asanyarray(depth_frame.get_data())
        	color_image = np.asanyarray(color_frame.get_data())
        	
        	# Apply colormap on depth image (image must be converted to 8-bit per pixel first
        	depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        	depth_colormap_dim = depth_colormap.shape
        	color_colormap_dim = color_image.shape
        	resized_color_image=color_image
        	if depth_colormap_dim != color_colormap_dim:
        		resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        	
        	frame=resized_color_image
        	
        	# Reading an image in default mode
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
        	result_color.write(frame)
        	cv2.imshow('mask', mask)
        	result_mask.write(mask)
        	cv2.imshow('result', result)
        	filter_result.write(result)
        	cv2.imshow("Depth Frame", depth_colormap)
        	result_depth.write(depth_colormap)
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
        	fig.canvas.draw()
        	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        	#plt.show()
        	plt.clf()
        	
        	img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        	# img is rgb, convert to opencv's default bgr
        	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        	
        	# display image and flip figure with flip code 0 for 180 mdegrees rotation of image
        	cv2.imshow("Source Ellipsoids",cv2.flip(img, 0))
        	result_ellipse.write(img)
        	
        	#waits for user to press any key 
        	#(this is necessary to avoid Python kernel form crashing)
        	if cv2.waitKey(1) == ord('q'):
        		break

finally:
	filter_result.release()
	result_mask.release()
	result_color.release()
	result_depth.release()
	result_ellipse.release()
	cv2.destroyAllWindows()
	# Stop streaming
	pipeline.stop()
