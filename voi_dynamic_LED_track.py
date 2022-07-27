#! /usr/bin/env python3
# The previous line will ensure the interpreter used is the first one on your environment's $PATH. Every Python file needs to start with this line at the top.

import rospy
import cv2 
import sep
from sensor_msgs.msg import Image as msg_Image
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Header
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from object_estimation.msg import ObjectsList_6state as msg_obj
from object_estimation.msg import tracked_object_xy as msg_xy
from geometry_msgs.msg import PoseWithCovarianceStamped 
from nav_msgs.msg import Odometry
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import autograd.numpy as np
from autograd import grad, jacobian
import tf
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import math
from matplotlib.patches import Ellipse
import pyrealsense2 as rs
import time


class GaussianObject: #this is an abstract class for repsesenting objects as PDFs: currently using multivariate Gaussians
	def __init__(self,z,m,T,Rot): #here z is a measurement <d,d_theta_x,d_theta_y>
		self.dim_x=6 #3-dimensional state space
		self.dim_z=3 #only observe the centroid position
		self.Rot=Rot #robot odometry rotation matrix relative to initial robot pose
		self.T=T #robot odometry translation vector to compute absolute positions in space
		self.m = m #object class ID
		
		self.censoring_level= 0.0 # CONSENSUS HYPERPARAMETER IN VoI
		self.x=np.array([[0],[0],[0],[0],[0],[0]])
		self.x = self.h_inverse(z) #3D state estimate, absolute object position and velocity vector in cartesian system
		self.xkk=self.x
		self.xkk_1=self.x
		
		self.Qk=0.0001*np.eye(self.dim_x) 
		self.Rk=np.array([[0.03,0.0,0.0],[0.0,0.04,0.0],[0.0,0.0,0.04]]) #sensor noise, mostly due to depth map and nonlinear propagation
		
		self.Pkk=0.0001*np.eye(self.dim_x) #initialize it as Q0, process noise covariance
		self.Pkk_1=self.Pkk
		
		self.update_counts=1 #how frequent you update this object position in space
		self.ykk = np.matmul(np.linalg.inv(self.Pkk),self.x)  #
		self.Ykk = np.linalg.inv(self.Pkk) #
		self.ykk_1 = self.ykk #
		self.Ykk_1 = self.Ykk  #
		
		self.z=z
		
		self.Y_est=self.Ykk_1 
		self.y_est=self.ykk_1
		self.Y_local=self.Ykk
		self.y_local=self.ykk 
		#self.time= #ROS time
		
		self.max_dt=1.5 #expressedn in seconds
		self.dt=0.5 # FOr a detector working at 2Hz this is the expected inter-measurement time
		self.last_time=time.time()
		
	def h_inverse(self,z): #state estimation from measurement
		d=z[0,0]
		d_theta_x=z[1,0]
		d_theta_y=z[2,0]
		vx=self.x[3,0];
		vy=self.x[4,0];
		vz=self.x[5,0];
		Dz=-d*np.sin(d_theta_x)
		Dy=-d*np.sin(d_theta_y)
		Dx=(d**2-Dz**2-Dy**2)**0.5
		rel_position_vector=np.array([[Dx],[Dy],[Dz]])
		abs_pos_vector=self.T+np.matmul(np.linalg.inv(self.Rot),rel_position_vector)
		estimated_state=np.array([[abs_pos_vector[0,0]],[abs_pos_vector[1,0]],[abs_pos_vector[2,0]],[vx],[vy],[vz]])
		return estimated_state
		
	def h(self, x): #observation model, predicted measurement given current state vector
		#here x is a column vector with 3 components, the current state estimate self.x
		# ONLY CENTROID POSITION IS VISIBLE
		x_rel=np.matmul(self.Rot,x[0:3,0]-self.T)
		
		xi=x_rel[0,0]
		yi=x_rel[1,0]
		zi=x_rel[2,0]
		#add translation due to velocity
		
		
		d=np.sqrt(xi**2+yi**2+zi**2)
		d_theta_x=-np.arcsin(zi/d)
		d_theta_y=-np.arcsin(yi/d)
		
		return np.array([[d],[d_theta_x],[d_theta_y]])	
	
	def value_of_information_decision(self):
		self.Y_est=self.Ykk_1 
		self.y_est=self.ykk_1
		self.Y_local=self.Ykk
		self.y_local=self.ykk 
		# Convert information to estimates
		self.x_est = np.linalg.inv(self.Y_est).dot(self.y_est)
		self.P_est = np.linalg.inv(self.Y_est)
		self.x_local = np.linalg.inv(self.Y_local).dot(self.y_local)
		self.P_local = np.linalg.inv(self.Y_local)        
		P_local_inv = np.linalg.inv(self.P_local)
		diff = self.x_local - self.x_est
		tmp1 = np.trace(P_local_inv.dot(self.P_est))
		tmp2 = diff.transpose().dot(P_local_inv).dot(diff)
		tmp3 = -self.dim_x + np.log(np.linalg.det(self.P_local)/np.linalg.det(self.P_est))
		kl_divergence = 0.5 * (tmp1 + tmp2 + tmp3) 
		if kl_divergence >= self.censoring_level:
		# Publish filter state
			return True
		else:
			return False
			
		
			
	def EKF_update(self,zk):
		dt=time.time()-self.last_time
		self.last_time=time.time()
		#zk is the new measurement vector, a column <d,Dx,Dy> of distance and angular displacements
		self.z=zk
		Rot=self.Rot.copy()
		T=self.T.copy()
		
		def h(x): #observation model, predicted measurement given current state vector
			#here x is a column vector with 6 components, the current state estimate self.x with position and velocity
	
			x_rel=np.matmul(Rot,x[0:3,0]-T)
	
			xi=x_rel[0,0]
			yi=x_rel[1,0]
			zi=x_rel[2,0]
			d=np.sqrt(xi**2+yi**2+zi**2)
			d_theta_x=-np.arcsin(zi/d)
			d_theta_y=-np.arcsin(yi/d)
			return np.array([[d],[d_theta_x],[d_theta_y]])
			
			
		#compute Jacobian matrices for linearizing system
		h_jacobian_operator=jacobian(h)
		if dt<self.max_dt:
			Ak=np.array([[1,0,0,self.dt,0,0],[0,1,0,0,self.dt,0],[0,0,1,0,0,self.dt],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
		else:
			Ak=np.eye(6)
		Fk=Ak #state dynamics for a static environment with A=I_3
		
		Hk=h_jacobian_operator(self.x).reshape(self.dim_z,self.dim_x)
		#a fe instructions to check values and shapes of matrices and vectors, for system block testing
		#print("current state is x="+str(self.x))
		#print("zk is"+str(zk))
		#print("Hk is"+str(Hk))
		#print("Hk shape is "+str(Hk.shape))
		#print("reshaped Hk is "+str(Hk.reshape(3,3)))
		Hk=Hk.reshape(self.dim_z,self.dim_x) #the jacobian must be a 3x3 matrix

		#EKF prediction step
		self.xkk_1=np.matmul(Fk,self.xkk)
		self.Pkk_1=np.matmul(Fk,np.matmul(self.Pkk,Fk.T))+self.Qk
		#EKF update step
		rez_yk=zk-self.h(self.xkk_1) # state innovation
		rez_Sk=np.matmul(Hk,np.matmul(self.Pkk_1,Hk.T))+self.Rk # covariance of state innovation
		#print("Pkk_1 is")
		#print(Pkk_1)
		#print("--------")
		#print("Jacobian of measurement, Hk is")
		#print(Hk)
		#print("------------")
		#print("Residual Sk is")
		#print(rez_Sk)
		#print("--------")
		#print("Inverse of Sk is")
		#print(np.linalg.inv(rez_Sk))
		#print("---------")
		Kk=np.matmul(self.Pkk_1,np.matmul(Hk.T,np.linalg.inv(rez_Sk))) #Kalman filter gain
		#print("Kalman gain is")
		#print(Kk)
		#print("---------")
		self.xkk=self.xkk_1+np.matmul(Kk,rez_yk)
		
		#print("xkk is")
		#print(xkk)
		#print("--------")
		self.Pkk=np.matmul(np.eye(self.dim_x)-np.matmul(Kk,Hk),self.Pkk_1)
		#print("Pkk is")
		#print(Pkk)
		#print("************END_OF_EKF_UPDATE********")
		#print()

		#Finnaly update these EKF filtered values for the object state and covariance
		self.x=self.xkk
		self.update_counts+=1
		#UPDATE INFORMATION MATRIX AND VECTOR REPRESENTATION
		
		self.ykk = np.matmul(np.linalg.inv(self.Pkk),self.xkk)  
		self.Ykk = np.linalg.inv(self.Pkk) #
		
		self.ykk_1 = np.matmul(np.linalg.inv(self.Pkk_1),self.xkk_1)  
		self.Ykk_1 = np.linalg.inv(self.Pkk_1) 





class ObjectPosEstimator:
	def __init__(self):
		self.bridge = CvBridge()
		rospy.init_node("nuc03_obj_pos_estimation")
		
		self.depth_cv_image=None
		self.color_image=None
		
		self.odom_list=[0,0,0,0,0,0,1]
		self.odom_list_absolute=[0,0,0,0,0,0,1]
		self.detection_number=0
		
		
		self.f = open("LED_tracker_measurements3.txt", 'a')
		self.f.write('****************NewMeasurementSession********************\n') 
		self.f.write('absolute <x,y,z> in meters, detection box Class ID, ROS time, Apparent Dx and Dy angular displacements (degrees)  \n') 
		
		self.f2 = open("LED_tracker_detection_data3.txt", 'a')
		self.f2.write('****************NewMeasurementSession********************\n') 
		self.f2.write('n,m,update_counts, time_now, x[0,0],x[1,0],x[2,0], Pkk[0,0],Pkk[0,1],Pkk[0,2],Pkk[1,0],Pkk[1,1],Pkk[1,2], Pkk[2,0],Pkk[2,1],Pkk[2,2] \n') 
		
		
		self.light_x=None
		self.light_y=None
		
		self.Obj_list=[] #this is the list with the detected object instances in the computer memory
		self.epsilon=1.5
		
		self.nuc01_objects=[]
		self.nuc02_objects=[]
		self.nuc03_objects=[]
		self.nuc04_objects=[]
		self.nuc05_objects=[]
		
		self.nuc03_consensus_objects=[] #this list will store data about VoI filtered objects
		
		rospy.Subscriber('/nuc03/d400/depth/image_rect_raw', msg_Image, self.imageDepthCallback)
		rospy.Subscriber('/nuc03/d400/color/image_raw', msg_Image , self.imageColorCallback)
		#rospy.Subscriber('/nuc01/t265/odom/sample', Odometry, self.odomcallback)
		
		
		#rospy.Subscriber('/nuc01/object_world', msg_obj, self.Distributed_VoI_filter_nuc01, queue_size=1) #subscribe to ObjList from Rover i
		rospy.Subscriber('/nuc02/object_world', msg_obj, self.Distributed_VoI_filter_nuc02, queue_size=1) #subscribe to ObjList from Rover i+1
		rospy.Subscriber('/nuc03/object_world', msg_obj, self.Distributed_VoI_filter_nuc03, queue_size=1) #subscribe to ObjList from Rover i+2 and so on
		rospy.Subscriber('/nuc04/object_world', msg_obj, self.Distributed_VoI_filter_nuc04, queue_size=1)
		rospy.Subscriber('/nuc05/object_world', msg_obj, self.Distributed_VoI_filter_nuc05, queue_size=1)
		
		self.rover_light_xy=rospy.Publisher('nuc03/rover_light_xy', msg_xy)
		self.pub=rospy.Publisher('nuc03/object_world',msg_obj)
		self.marker_publisher = rospy.Publisher('nuc03/consensus_objects_markers', MarkerArray)
		
		
		
		listener=tf.TransformListener()
		while not rospy.is_shutdown():
			try:
				(trans,rot) = listener.lookupTransform('map', 'nuc03/t265_pose_frame', rospy.Time(0))
				t_x = trans[0]
				t_y = trans[1]
				t_z = trans[2]
				q_x = rot[0]
				q_y = rot[1]
				q_z = rot[2]
				q_w = rot[3]
				self.odom_list_absolute=[t_x,t_y,t_z,q_x,q_y,q_z,q_w]
				#print(self.odom_list_absolute)
			except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				continue
		
		rospy.spin()
	
	
	
	
	
	def imageDepthCallback(self, data):
		try:
			self.depth_cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
			height= len(self.depth_cv_image[0])
			width = len(self.depth_cv_image[1])
			#print("Width is: "+str(int(width)))
			#print("Height is: "+str(int(height)))
			#print(cv_image.shape)
			#print(cv_image[height//2,width//2])
		except CvBridgeError as e:
			print(e)
			return
			
			
			  
	def imageColorCallback(self, data):
		self.color_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
		
		# Apply colormap on depth image (image must be converted to 8-bit per pixel first
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_cv_image, alpha=0.03), cv2.COLORMAP_JET)
		depth_colormap_dim = depth_colormap.shape
		color_colormap_dim = self.color_image.shape
		resized_color_image=self.color_image
		if depth_colormap_dim != color_colormap_dim:
			resized_color_image = cv2.resize(self.color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
		frame=resized_color_image
		# Reading an image in default mode
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#cv2.imshow("Image Window", frame)
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
		#cv2.imshow('frame', frame)
		#cv2.imshow('mask', mask)
		#cv2.imshow('result', result)
		#cv2.imshow("Depth Frame", depth_colormap)
		# plot background-subtracted image
		#fig, ax = plt.subplots()
		#m, s = np.mean(gray_result), np.std(gray_result)
		#im = ax.imshow(gray_result, interpolation='nearest', cmap='gray',vmin=m-s, vmax=m+s, origin='lower')
		# plot an ellipse for each object
		brightest_id=None #search the objects list for the brightest object available, which could be the tracked object
		for i in range(len(objects)):
			e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
			width=6*objects['a'][i],
			height=6*objects['b'][i],
			angle=objects['theta'][i] * 180. / np.pi)
			e.set_facecolor('none')
			e.set_edgecolor('red')
			#ax.add_artist(e)
			if brightest_id==None:
				brightest_id=i
			else:
				if objects['a'][i]*objects['b'][i]>objects['a'][brightest_id]*objects['b'][brightest_id]:
					brightest_id=i
		#print("Brightest Object < x,y > = <" + str(objects['x'][brightest_id])+", " + str(objects['y'][brightest_id]) + ">")
		xy_message = msg_xy()
		xy_message.x=objects['x'][brightest_id]
		xy_message.y=objects['y'][brightest_id]
		if(len(objects)>0):
			self.rover_light_xy.publish(xy_message)
		self.light_x=objects['x'][brightest_id]
		self.light_y=objects['y'][brightest_id]
		
		#print("********************")
		#print("New Detection Series\n\n")
		#print("Absolute Position (meters) and Quaternion of Robot : "+str(self.odom_list_absolute))
		ok=1
		# Generate a rotation operation object using the rotation quaternion
		r = R.from_quat([self.odom_list_absolute[3], self.odom_list_absolute[4], self.odom_list_absolute[5], self.odom_list_absolute[6]])
		rotation_matrix=r.as_matrix() # robot initial pose system to the current pose reference system
		
		
		Rx=640 #cv_image.shape[1]
		Ry=480 #cv_image.shape[0]
		FOVx=69 #degrees, RGB IntelRealsense D435
		FOVy=42 #degrees, RGB IntelRealsense D435
		# calculate angular displacements on x and y axis, knowing the camera FOV (for color frame)
		
		d_theta_x=((self.light_x-Rx)*FOVx/2)/Rx
		d_theta_y=((self.light_y-Ry)*FOVy/2)/Ry
		
		# estimate scalar distance to detected object centroid, in mm
		d=self.depth_cv_image[int(self.light_x),int(self.light_y)]
		#print("Brightest point x = "+str(int(self.light_x)))
		#print("Brightest point y = "+str(int(self.light_y)))
		#transform the distance to meters (International System Unit)
		d=d*0.001
		d+=0.085
		if d>0 and d<5: # proceed further with this distance estimate only s D435 gave a valid measurement result
			if ok==1:
				ok=0
				self.detection_number+=1 #this is a new valid detection
			#print("Distance to object centroid is, in meters: "+str(d))
			# Estimate relative position vector of the detected object, measured from the robot's D435 origin
			Dz=-d*np.sin(np.deg2rad(d_theta_x))
			Dy=-d*np.sin(np.deg2rad(d_theta_y))
			Dx=(d**2-Dz**2-Dy**2)**0.5
			d_theta_x=np.deg2rad(d_theta_x)
			d_theta_y=np.deg2rad(d_theta_y)
			#print("Relative Position vector, in meters <x,y,z> = <"+str(Dx)+","+str(Dy)+","+str(Dz)+">")
			rel_position_vector=np.array([[Dx],[Dy],[Dz]])
			robot_position_vector=np.array([[self.odom_list_absolute[0]],[self.odom_list_absolute[1]],[self.odom_list_absolute[2]]])
			# Calculate the absolute position of the object in space
			abs_position_vector=robot_position_vector+np.matmul(np.linalg.inv(rotation_matrix),rel_position_vector)
			#print("Absolute Object Centroid Position (meters) <x,y,z>: <"+str(abs_position_vector[0,0])+", "+str(abs_position_vector[1,0])+", "+str(abs_position_vector[2,0]))
			#print("\n")
			
			data_string=''
			data_string+=str(abs_position_vector[0,0])+","+str(abs_position_vector[1,0])+","+str(abs_position_vector[2,0])
			#data_string+=","+str(box.id)+","+ str(rospy.Time.now())+","+str(self.detection_number)+","+str(d_theta_x)+","+str(d_theta_y)
			self.f.write(data_string+'\n')
			
			#put NewObject and EKF update code here
			z=np.array([[d],[d_theta_x],[d_theta_y]])
			m=100 #arbitrarily chosen class id for the green light
			Rot=rotation_matrix
			T=robot_position_vector
			if len(self.Obj_list)==0: #when robot starts adding objects in memory, but currently has none
				self.Obj_list.append(GaussianObject(z,m,T,Rot))
				if np.linalg.norm(self.Obj_list[0].x[2,0])>self.epsilon: # rovers lights should have relatively low absolute values of the z coordinate
					self.Obj_list=[]
				#print(GaussianObject(z,m).x-np.array([[Dx],[Dy],[Dz]])) #there should be no difference
			else: #there is already an object in the robot memory,try matching them 
				obj=self.Obj_list[0]
				if np.linalg.norm(obj.x[2,0]-abs_position_vector[2,0])<self.epsilon: #altitude filter to neglect lights 
					#do the EKF update and update the current object Rotation and Translation
					obj.Rot=Rot;obj.T=T
					obj.EKF_update(z) #before any EKF update aslo update T and R matrices of odometry
							
					
		#else:			
			#print("D435 distance estimate was innacurate")
			#print("\n")
		# Estimate relative position vector of nthe detected object, measured from the robot's D435 origin
		#print("\n")
		#print("There are n="+str(len(self.Obj_list))+" objects in my robot memory!")
		
			#except Exception as e:
			#print(e)
			#return
		#construct the ROS message payload here and send it
		msg=msg_obj()
		n=0
		#iterate over detected objects and append data to the msg
		for obj in self.nuc03_consensus_objects:
			msg.class_ids.append(obj.m) #add the detected object class id in the ROS message
			#create a PoseStampedWithCovariance message
			PoseStampedWithCovariance=PoseWithCovarianceStamped() #enter data for the 
			PoseStampedWithCovariance.pose.pose.position.x=obj.x[0,0]
			PoseStampedWithCovariance.pose.pose.position.y=obj.x[1,0]
			PoseStampedWithCovariance.pose.pose.position.z=obj.x[2,0]
			PoseStampedWithCovariance.pose.pose.orientation.x=0 #rotations of objects in space are irrelevant: state null rotation
			PoseStampedWithCovariance.pose.pose.orientation.y=0
			PoseStampedWithCovariance.pose.pose.orientation.z=0
			PoseStampedWithCovariance.pose.pose.orientation.w=1
			PoseStampedWithCovariance.pose.covariance[0]=obj.Pkk[0,0] #float36 covariance of the object PDF, diagonal is [0], [7], [14]
			PoseStampedWithCovariance.pose.covariance[1]=obj.Pkk[0,1]
			PoseStampedWithCovariance.pose.covariance[2]=obj.Pkk[0,2]
			PoseStampedWithCovariance.pose.covariance[7]=obj.Pkk[1,0]
			PoseStampedWithCovariance.pose.covariance[8]=obj.Pkk[1,1]
			PoseStampedWithCovariance.pose.covariance[9]=obj.Pkk[1,2]
			PoseStampedWithCovariance.pose.covariance[14]=obj.Pkk[2,0]
			PoseStampedWithCovariance.pose.covariance[15]=obj.Pkk[2,1]
			PoseStampedWithCovariance.pose.covariance[16]=obj.Pkk[2,2]
			msg.poses.append(PoseStampedWithCovariance)
			#create a PoseStampedWithCovariance message
			PoseStampedWithCovariance_dot=PoseWithCovarianceStamped() #enter data for the 
			PoseStampedWithCovariance_dot.pose.pose.position.x=obj.x[3,0]
			PoseStampedWithCovariance_dot.pose.pose.position.y=obj.x[4,0]
			PoseStampedWithCovariance_dot.pose.pose.position.z=obj.x[5,0]
			PoseStampedWithCovariance_dot.pose.pose.orientation.x=0 #rotations of objects in space are irrelevant: state null rotation
			PoseStampedWithCovariance_dot.pose.pose.orientation.y=0
			PoseStampedWithCovariance_dot.pose.pose.orientation.z=0
			PoseStampedWithCovariance_dot.pose.pose.orientation.w=1
			PoseStampedWithCovariance_dot.pose.covariance[0]=obj.Pkk[3,3] #float36 covariance of the object PDF, diagonal is [0], [7], [14]
			PoseStampedWithCovariance_dot.pose.covariance[1]=obj.Pkk[3,4]
			PoseStampedWithCovariance_dot.pose.covariance[2]=obj.Pkk[3,5]
			PoseStampedWithCovariance_dot.pose.covariance[7]=obj.Pkk[4,3]
			PoseStampedWithCovariance_dot.pose.covariance[8]=obj.Pkk[4,4]
			PoseStampedWithCovariance_dot.pose.covariance[9]=obj.Pkk[4,5]
			PoseStampedWithCovariance_dot.pose.covariance[14]=obj.Pkk[5,3]
			PoseStampedWithCovariance_dot.pose.covariance[15]=obj.Pkk[5,4]
			PoseStampedWithCovariance_dot.pose.covariance[16]=obj.Pkk[5,5]
			msg.poses_dot.append(PoseStampedWithCovariance)
		#rospy.loginfo(msg)
		self.pub.publish(msg)
		
		self.nuc03_consensus_objects=[]
		#CONSENSUS BASED ON LOGOP AND KL DIVERGENCE BEGINS HERE
		
		#for each object from self.Obj_list try matching with objects from self.nuc0X_objects=[] where x is not this NUC's number
		# iterate over self.Obj_list instead of self.nuc01
		for obj in self.Obj_list:
			best_match_from_nuc03=obj #this code runs on nuc01
			best_match1=None
			for obj_possible_match in self.nuc01_objects:
				if np.linalg.norm(obj.x-obj_possible_match.x)<self.epsilon and obj.m==obj_possible_match.m and len(self.nuc01_objects)>0:
					if best_match1==None:
						best_match1=obj_possible_match #the best match is an abstract object
					else:
						#compare best_match and the current obj_possible_match
						if np.linalg.norm(obj.x-obj_possible_match.x)<np.linalg.norm(obj.x-best_match1.x): #check euclidian distance
							best_match1=obj_possible_match
			
			#at this point you know what is the match for obj (from self.Obj_list) from the self.nuc02_objects
			best_match_from_nuc01=best_match1
			
			#consider doing LogOP with nuc01 and nuc02
			#if Dkl criteria do the aggregation - IMPLEMENT KL DIVERGENCE ESTIMATION
			if best_match_from_nuc03.value_of_information_decision()==1 and len(self.nuc01_objects)>0: # if KL divergence between prediction and measured state is high enough
				if best_match_from_nuc01!=None: #do LogOP aggregation
					ykk1 = np.matmul(np.linalg.inv(best_match_from_nuc01.Pkk),best_match_from_nuc01.x)  #
					Ykk1 = np.linalg.inv(best_match_from_nuc01.Pkk) #
					ykk3 = np.matmul(np.linalg.inv(best_match_from_nuc03.Pkk),best_match_from_nuc03.x)  #
					Ykk3 = np.linalg.inv(best_match_from_nuc03.Pkk) #
					
					ykk=(ykk1+ykk3)/np.sqrt(2) #2 = cardinal(numbers of rovers in local network)
					Ykk=(Ykk1+Ykk3)/np.sqrt(2)
					
					#now after doing logOP consensus (for two rovers) create a new object instance
					aggregated_obj=GaussianObject(best_match_from_nuc01.z,best_match_from_nuc01.m,best_match_from_nuc01.T,best_match_from_nuc01.Rot)
					aggregated_obj.ykk=ykk
					aggregated_obj.Ykk=Ykk
					aggregated_obj.Y_local=Ykk
					aggregated_obj.y_local=ykk
					#consensus (ykk,Ykk)-->(x,Pkk) updated PDF based on nonlinear distributed estimation 
					aggregated_obj.xkk = np.linalg.inv(aggregated_obj.Y_local).dot(aggregated_obj.y_local)
					aggregated_obj.Pkk = np.linalg.inv(aggregated_obj.Y_local) 
					
					self.nuc03_consensus_objects.append(aggregated_obj)
				else: 
					self.nuc03_consensus_objects.append(obj) #single agent EKF estimation works best, KL divergence is small
			else: 
				self.nuc03_consensus_objects.append(obj) #single agent EKF estimation works best, KL divergence is small
					
			#PUBLISH POSEWITHCOVARIANCE MESSAGE USING self.nuc01_consensus_objects FOR USE IN RVIZ
		
			
			
		# CONSENSUS ALGORITHM END
		print("************************")
		obj=self.nuc03_consensus_objects[0]
		print(obj.Pkk[1,1])
		
		
		
		#PUBLISH POSEWITHCOVARIANCE MESSAGE USING self.nuc01_consensus_objects FOR USE IN RVIZ
		markerArray = MarkerArray()
		for obj in self.nuc03_consensus_objects:
			
			marker = Marker()
			marker.header.frame_id = "map"
			marker.type = marker.SPHERE
			marker.action = marker.ADD
			marker.scale.x = 0.2 #0.2
			marker.scale.y = 0.2 #0.2
			marker.scale.z = 0.2 #0.2
			marker.color.a = 1.0
			marker.color.r = 1.0
			marker.color.g = 1.0
			marker.color.b = 0.0
			marker.pose.orientation.w = 1.0
			marker.pose.position.x = obj.x[0,0]
			marker.pose.position.y = obj.x[1,0]
			marker.pose.position.z = obj.x[2,0]
			markerArray.markers.append(marker)
		
		
		# Give the marker IDs
		marker_id = 0
		for m in markerArray.markers:
			m.id = marker_id
			marker_id += 1
		
		
		self.marker_publisher.publish(markerArray)
		
		n=0
		data_string2=''
		for obj in self.nuc03_consensus_objects:
			n+=1
			data_string2+=str(n)+',' #order index of object in the list
			data_string2+=str(obj.m)+','
			data_string2+=str(obj.update_counts)+','
			data_string2+=str(time.time())+','
			data_string2+=str(obj.x[0,0])+','
			data_string2+=str(obj.x[1,0])+','
			data_string2+=str(obj.x[2,0])+','
			data_string2+=str(obj.Pkk[0,0])+','
			data_string2+=str(obj.Pkk[0,1])+','
			data_string2+=str(obj.Pkk[0,2])+','
			data_string2+=str(obj.Pkk[1,0])+','
			data_string2+=str(obj.Pkk[1,1])+','
			data_string2+=str(obj.Pkk[1,2])+','
			data_string2+=str(obj.Pkk[2,0])+','
			data_string2+=str(obj.Pkk[2,1])+','
			data_string2+=str(obj.Pkk[2,2])+'\n'
		if data_string2!='':
			self.f2.write(data_string2)
		
		
		
	def Distributed_VoI_filter_nuc01(self, data):
		self.nuc01_objects=[]
		#rospy.loginfo("VOI ALGORITHM CALLBACK NUC01")
		
		m_list=data.class_ids #the list of object class indices
		object_poses_list=data.poses
		object_poses_dot_list=data.poses_dot
		new_objects_list=[]
		for i in range(len(m_list)):
			m=m_list[i]
			Pkk=np.zeros((6,6)) #covarianta
			stampedposewithcovariance=object_poses_list[i]
			stampedposewithcovariance_dot=object_poses_dot_list[i]
			x=np.array([[0],[0],[0],[0],[0],[0]]) #pozitia
			x[0,0]=stampedposewithcovariance.pose.pose.position.x #verifica daca datele din vectorul pozitiei absolute sunt updatate corect
			x[1,0]=stampedposewithcovariance.pose.pose.position.y
			x[2,0]=stampedposewithcovariance.pose.pose.position.z
			Pkk[0,0]=stampedposewithcovariance.pose.covariance[0]
			Pkk[0,1]=stampedposewithcovariance.pose.covariance[1]
			Pkk[0,2]=stampedposewithcovariance.pose.covariance[2]
			Pkk[1,0]=stampedposewithcovariance.pose.covariance[7]
			Pkk[1,1]=stampedposewithcovariance.pose.covariance[8]
			Pkk[1,2]=stampedposewithcovariance.pose.covariance[9]
			Pkk[2,0]=stampedposewithcovariance.pose.covariance[14]
			Pkk[2,1]=stampedposewithcovariance.pose.covariance[15]
			Pkk[2,2]=stampedposewithcovariance.pose.covariance[16]
			x[3,0]=stampedposewithcovariance_dot.pose.pose.position.x
			x[4,0]=stampedposewithcovariance_dot.pose.pose.position.y
			x[5,0]=stampedposewithcovariance_dot.pose.pose.position.z
			Pkk[3,3]=stampedposewithcovariance_dot.pose.covariance[0]
			Pkk[3,4]=stampedposewithcovariance_dot.pose.covariance[1]
			Pkk[3,5]=stampedposewithcovariance_dot.pose.covariance[2]
			Pkk[4,3]=stampedposewithcovariance_dot.pose.covariance[7]
			Pkk[4,4]=stampedposewithcovariance_dot.pose.covariance[8]
			Pkk[4,5]=stampedposewithcovariance_dot.pose.covariance[9]
			Pkk[5,3]=stampedposewithcovariance_dot.pose.covariance[14]
			Pkk[5,4]=stampedposewithcovariance_dot.pose.covariance[15]
			Pkk[5,5]=stampedposewithcovariance_dot.pose.covariance[16]
			#rospy.loginfo(str(x))
			#rospy.loginfo(str(Pkk))
			#rospy.loginfo(str(m))
			obj=GaussianObject(np.array([[0],[0],[0]]),m,np.array([[0],[0],[0]]),np.eye(3))
			obj.x=x
			obj.Pkk=Pkk
			obj.ykk = np.matmul(np.linalg.inv(obj.Pkk),obj.x)  #
			obj.Ykk = np.linalg.inv(obj.Pkk) #
			obj.ykk_1 = obj.ykk #
			obj.Ykk_1 = obj.Ykk  #
			
			obj.Y_est=obj.Ykk_1 
			obj.y_est=obj.ykk_1
			obj.Y_local=obj.Ykk
			obj.y_local=obj.ykk 
			obj.dim_x=3 #3-dimensional state space
			new_objects_list.append(obj)
			self.nuc01_objects.append(obj)
		
		
	def Distributed_VoI_filter_nuc02(self, data):
		self.nuc02_objects=[]
		#rospy.loginfo("VOI ALGORITHM CALLBACK NUC02")
		
		m_list=data.class_ids #the list of object class indices
		object_poses_list=data.poses
		object_poses_dot_list=data.poses_dot
		new_objects_list=[]
		for i in range(len(m_list)):
			m=m_list[i]
			Pkk=np.zeros((6,6)) #covarianta
			stampedposewithcovariance=object_poses_list[i]
			stampedposewithcovariance_dot=object_poses_dot_list[i]
			x=np.array([[0],[0],[0],[0],[0],[0]]) #pozitia
			x[0,0]=stampedposewithcovariance.pose.pose.position.x #verifica daca datele din vectorul pozitiei absolute sunt updatate corect
			x[1,0]=stampedposewithcovariance.pose.pose.position.y
			x[2,0]=stampedposewithcovariance.pose.pose.position.z
			Pkk[0,0]=stampedposewithcovariance.pose.covariance[0]
			Pkk[0,1]=stampedposewithcovariance.pose.covariance[1]
			Pkk[0,2]=stampedposewithcovariance.pose.covariance[2]
			Pkk[1,0]=stampedposewithcovariance.pose.covariance[7]
			Pkk[1,1]=stampedposewithcovariance.pose.covariance[8]
			Pkk[1,2]=stampedposewithcovariance.pose.covariance[9]
			Pkk[2,0]=stampedposewithcovariance.pose.covariance[14]
			Pkk[2,1]=stampedposewithcovariance.pose.covariance[15]
			Pkk[2,2]=stampedposewithcovariance.pose.covariance[16]
			x[3,0]=stampedposewithcovariance_dot.pose.pose.position.x
			x[4,0]=stampedposewithcovariance_dot.pose.pose.position.y
			x[5,0]=stampedposewithcovariance_dot.pose.pose.position.z
			Pkk[3,3]=stampedposewithcovariance_dot.pose.covariance[0]
			Pkk[3,4]=stampedposewithcovariance_dot.pose.covariance[1]
			Pkk[3,5]=stampedposewithcovariance_dot.pose.covariance[2]
			Pkk[4,3]=stampedposewithcovariance_dot.pose.covariance[7]
			Pkk[4,4]=stampedposewithcovariance_dot.pose.covariance[8]
			Pkk[4,5]=stampedposewithcovariance_dot.pose.covariance[9]
			Pkk[5,3]=stampedposewithcovariance_dot.pose.covariance[14]
			Pkk[5,4]=stampedposewithcovariance_dot.pose.covariance[15]
			Pkk[5,5]=stampedposewithcovariance_dot.pose.covariance[16]
			#rospy.loginfo(str(x))
			#rospy.loginfo(str(Pkk))
			#rospy.loginfo(str(m))
			obj=GaussianObject(np.array([[0],[0],[0]]),m,np.array([[0],[0],[0]]),np.eye(3))
			obj.x=x
			obj.Pkk=Pkk
			obj.ykk = np.matmul(np.linalg.inv(obj.Pkk),obj.x)  #
			obj.Ykk = np.linalg.inv(obj.Pkk) #
			obj.ykk_1 = obj.ykk #
			obj.Ykk_1 = obj.Ykk  #
			
			obj.Y_est=obj.Ykk_1 
			obj.y_est=obj.ykk_1
			obj.Y_local=obj.Ykk
			obj.y_local=obj.ykk 
			obj.dim_x=3 #3-dimensional state space
			new_objects_list.append(obj)
			self.nuc02_objects.append(obj)
		
		
	def Distributed_VoI_filter_nuc03(self, data):
		self.nuc03_objects=[]
		#rospy.loginfo("VOI ALGORITHM CALLBACK NUC03")
		
		m_list=data.class_ids #the list of object class indices
		object_poses_list=data.poses
		object_poses_dot_list=data.poses_dot
		new_objects_list=[]
		for i in range(len(m_list)):
			m=m_list[i]
			Pkk=np.zeros((6,6)) #covarianta
			stampedposewithcovariance=object_poses_list[i]
			stampedposewithcovariance_dot=object_poses_dot_list[i]
			x=np.array([[0],[0],[0],[0],[0],[0]]) #pozitia
			x[0,0]=stampedposewithcovariance.pose.pose.position.x #verifica daca datele din vectorul pozitiei absolute sunt updatate corect
			x[1,0]=stampedposewithcovariance.pose.pose.position.y
			x[2,0]=stampedposewithcovariance.pose.pose.position.z
			Pkk[0,0]=stampedposewithcovariance.pose.covariance[0]
			Pkk[0,1]=stampedposewithcovariance.pose.covariance[1]
			Pkk[0,2]=stampedposewithcovariance.pose.covariance[2]
			Pkk[1,0]=stampedposewithcovariance.pose.covariance[7]
			Pkk[1,1]=stampedposewithcovariance.pose.covariance[8]
			Pkk[1,2]=stampedposewithcovariance.pose.covariance[9]
			Pkk[2,0]=stampedposewithcovariance.pose.covariance[14]
			Pkk[2,1]=stampedposewithcovariance.pose.covariance[15]
			Pkk[2,2]=stampedposewithcovariance.pose.covariance[16]
			x[3,0]=stampedposewithcovariance_dot.pose.pose.position.x
			x[4,0]=stampedposewithcovariance_dot.pose.pose.position.y
			x[5,0]=stampedposewithcovariance_dot.pose.pose.position.z
			Pkk[3,3]=stampedposewithcovariance_dot.pose.covariance[0]
			Pkk[3,4]=stampedposewithcovariance_dot.pose.covariance[1]
			Pkk[3,5]=stampedposewithcovariance_dot.pose.covariance[2]
			Pkk[4,3]=stampedposewithcovariance_dot.pose.covariance[7]
			Pkk[4,4]=stampedposewithcovariance_dot.pose.covariance[8]
			Pkk[4,5]=stampedposewithcovariance_dot.pose.covariance[9]
			Pkk[5,3]=stampedposewithcovariance_dot.pose.covariance[14]
			Pkk[5,4]=stampedposewithcovariance_dot.pose.covariance[15]
			Pkk[5,5]=stampedposewithcovariance_dot.pose.covariance[16]
			#rospy.loginfo(str(x))
			#rospy.loginfo(str(Pkk))
			#rospy.loginfo(str(m))
			obj=GaussianObject(np.array([[0],[0],[0]]),m,np.array([[0],[0],[0]]),np.eye(3))
			obj.x=x
			obj.Pkk=Pkk
			obj.ykk = np.matmul(np.linalg.inv(obj.Pkk),obj.x)  #
			obj.Ykk = np.linalg.inv(obj.Pkk) #
			obj.ykk_1 = obj.ykk #
			obj.Ykk_1 = obj.Ykk  #
			
			obj.Y_est=obj.Ykk_1 
			obj.y_est=obj.ykk_1
			obj.Y_local=obj.Ykk
			obj.y_local=obj.ykk 
			obj.dim_x=3 #3-dimensional state space
			new_objects_list.append(obj)
			self.nuc03_objects.append(obj)

	def Distributed_VoI_filter_nuc04(self, data):
		self.nuc04_objects=[]
		#rospy.loginfo("VOI ALGORITHM CALLBACK NUC04")
		
		m_list=data.class_ids #the list of object class indices
		object_poses_list=data.poses
		object_poses_dot_list=data.poses_dot
		new_objects_list=[]
		for i in range(len(m_list)):
			m=m_list[i]
			Pkk=np.zeros((6,6)) #covarianta
			stampedposewithcovariance=object_poses_list[i]
			stampedposewithcovariance_dot=object_poses_dot_list[i]
			x=np.array([[0],[0],[0],[0],[0],[0]]) #pozitia
			x[0,0]=stampedposewithcovariance.pose.pose.position.x #verifica daca datele din vectorul pozitiei absolute sunt updatate corect
			x[1,0]=stampedposewithcovariance.pose.pose.position.y
			x[2,0]=stampedposewithcovariance.pose.pose.position.z
			Pkk[0,0]=stampedposewithcovariance.pose.covariance[0]
			Pkk[0,1]=stampedposewithcovariance.pose.covariance[1]
			Pkk[0,2]=stampedposewithcovariance.pose.covariance[2]
			Pkk[1,0]=stampedposewithcovariance.pose.covariance[7]
			Pkk[1,1]=stampedposewithcovariance.pose.covariance[8]
			Pkk[1,2]=stampedposewithcovariance.pose.covariance[9]
			Pkk[2,0]=stampedposewithcovariance.pose.covariance[14]
			Pkk[2,1]=stampedposewithcovariance.pose.covariance[15]
			Pkk[2,2]=stampedposewithcovariance.pose.covariance[16]
			x[3,0]=stampedposewithcovariance_dot.pose.pose.position.x
			x[4,0]=stampedposewithcovariance_dot.pose.pose.position.y
			x[5,0]=stampedposewithcovariance_dot.pose.pose.position.z
			Pkk[3,3]=stampedposewithcovariance_dot.pose.covariance[0]
			Pkk[3,4]=stampedposewithcovariance_dot.pose.covariance[1]
			Pkk[3,5]=stampedposewithcovariance_dot.pose.covariance[2]
			Pkk[4,3]=stampedposewithcovariance_dot.pose.covariance[7]
			Pkk[4,4]=stampedposewithcovariance_dot.pose.covariance[8]
			Pkk[4,5]=stampedposewithcovariance_dot.pose.covariance[9]
			Pkk[5,3]=stampedposewithcovariance_dot.pose.covariance[14]
			Pkk[5,4]=stampedposewithcovariance_dot.pose.covariance[15]
			Pkk[5,5]=stampedposewithcovariance_dot.pose.covariance[16]
			#rospy.loginfo(str(x))
			#rospy.loginfo(str(Pkk))
			#rospy.loginfo(str(m))
			obj=GaussianObject(np.array([[0],[0],[0]]),m,np.array([[0],[0],[0]]),np.eye(3))
			obj.x=x
			obj.Pkk=Pkk
			obj.ykk = np.matmul(np.linalg.inv(obj.Pkk),obj.x)  #
			obj.Ykk = np.linalg.inv(obj.Pkk) #
			obj.ykk_1 = obj.ykk #
			obj.Ykk_1 = obj.Ykk  #
			
			obj.Y_est=obj.Ykk_1 
			obj.y_est=obj.ykk_1
			obj.Y_local=obj.Ykk
			obj.y_local=obj.ykk 
			obj.dim_x=3 #3-dimensional state space
			new_objects_list.append(obj)
			self.nuc04_objects.append(obj)
		

	def Distributed_VoI_filter_nuc05(self, data):
		self.nuc05_objects=[]
		#rospy.loginfo("VOI ALGORITHM CALLBACK NUC05")
		
		m_list=data.class_ids #the list of object class indices
		object_poses_list=data.poses
		object_poses_dot_list=data.poses_dot
		new_objects_list=[]
		for i in range(len(m_list)):
			m=m_list[i]
			Pkk=np.zeros((6,6)) #covarianta
			stampedposewithcovariance=object_poses_list[i]
			stampedposewithcovariance_dot=object_poses_dot_list[i]
			x=np.array([[0],[0],[0],[0],[0],[0]]) #pozitia
			x[0,0]=stampedposewithcovariance.pose.pose.position.x #verifica daca datele din vectorul pozitiei absolute sunt updatate corect
			x[1,0]=stampedposewithcovariance.pose.pose.position.y
			x[2,0]=stampedposewithcovariance.pose.pose.position.z
			Pkk[0,0]=stampedposewithcovariance.pose.covariance[0]
			Pkk[0,1]=stampedposewithcovariance.pose.covariance[1]
			Pkk[0,2]=stampedposewithcovariance.pose.covariance[2]
			Pkk[1,0]=stampedposewithcovariance.pose.covariance[7]
			Pkk[1,1]=stampedposewithcovariance.pose.covariance[8]
			Pkk[1,2]=stampedposewithcovariance.pose.covariance[9]
			Pkk[2,0]=stampedposewithcovariance.pose.covariance[14]
			Pkk[2,1]=stampedposewithcovariance.pose.covariance[15]
			Pkk[2,2]=stampedposewithcovariance.pose.covariance[16]
			x[3,0]=stampedposewithcovariance_dot.pose.pose.position.x
			x[4,0]=stampedposewithcovariance_dot.pose.pose.position.y
			x[5,0]=stampedposewithcovariance_dot.pose.pose.position.z
			Pkk[3,3]=stampedposewithcovariance_dot.pose.covariance[0]
			Pkk[3,4]=stampedposewithcovariance_dot.pose.covariance[1]
			Pkk[3,5]=stampedposewithcovariance_dot.pose.covariance[2]
			Pkk[4,3]=stampedposewithcovariance_dot.pose.covariance[7]
			Pkk[4,4]=stampedposewithcovariance_dot.pose.covariance[8]
			Pkk[4,5]=stampedposewithcovariance_dot.pose.covariance[9]
			Pkk[5,3]=stampedposewithcovariance_dot.pose.covariance[14]
			Pkk[5,4]=stampedposewithcovariance_dot.pose.covariance[15]
			Pkk[5,5]=stampedposewithcovariance_dot.pose.covariance[16]
			#rospy.loginfo(str(x))
			#rospy.loginfo(str(Pkk))
			#rospy.loginfo(str(m))
			obj=GaussianObject(np.array([[0],[0],[0]]),m,np.array([[0],[0],[0]]),np.eye(3))
			obj.x=x
			obj.Pkk=Pkk
			obj.ykk = np.matmul(np.linalg.inv(obj.Pkk),obj.x)  #
			obj.Ykk = np.linalg.inv(obj.Pkk) #
			obj.ykk_1 = obj.ykk #
			obj.Ykk_1 = obj.Ykk  #
			
			obj.Y_est=obj.Ykk_1 
			obj.y_est=obj.ykk_1
			obj.Y_local=obj.Ykk
			obj.y_local=obj.ykk 
			obj.dim_x=3 #3-dimensional state space
			new_objects_list.append(obj)
			self.nuc05_objects.append(obj)
			








	def odomcallback(self,data):
     		t_x = data.pose.pose.position.x
     		t_y = data.pose.pose.position.y
     		t_z = data.pose.pose.position.z
     		q_x = data.pose.pose.orientation.x
     		q_y = data.pose.pose.orientation.y
     		q_z = data.pose.pose.orientation.z 
     		q_w = data.pose.pose.orientation.w
     		self.odom_list=[t_x,t_y,t_z,q_x,q_y,q_z,q_w]
     		#print("Absolute Robot Position: "+str(t_x)+", "+str(t_y)+", "+str(t_z))
     		


if __name__ == '__main__':
	ObjectPosEstimator = ObjectPosEstimator()
	#rate=rospy.Rate(10)
	#while not rospy.is_shutdown():
		#rate.sleep()
		#print("ROSPY SPINNING")
	
	# Create an object to read 
	# from camera



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
