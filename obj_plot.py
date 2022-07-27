import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

data_file = open('obj_data.txt', 'r')
Lines = data_file.readlines()
data_list=[]
count = 0
# Strips the newline character
for line in Lines:
	count += 1
	#print(line.split(","))
	#print(type(line.split(",")))
	if count>2:
		#store data in the data list only after reading the first two lines of the text file *****NewMeasurement**** and x,y,x...
		data_list.append(line.split(","))
		aux=data_list[-1][-1]
		data_list[-1][-1]=aux[0:-1] #do this to eliminate special character \n from the last element in the list
#generate a single np.array with all the relevant data, and with 7 columns x,y,x,ID,ROStime,Dx,Dy
data_array=np.zeros((len(data_list),7)) 
for row_index in range(len(data_list)):
	data_array[row_index,0]=float(data_list[row_index][0]) #x
	data_array[row_index,1]=float(data_list[row_index][1]) #y
	data_array[row_index,2]=float(data_list[row_index][2]) #z
	data_array[row_index,3]=int(data_list[row_index][3]) #id
	data_array[row_index,4]=float(data_list[row_index][4]) #ROS_time
	data_array[row_index,5]=float(data_list[row_index][5]) #Dx degrees
	data_array[row_index,6]=float(data_list[row_index][6]) #Dy degrees

#what objects have been observed in this dataset?
id_list=[]
for row_index in range(len(data_list)):
	if data_array[row_index,3] not in id_list:
		id_list.append(int(data_array[row_index,3]))

#generate a dictionary with keys that are classes IDs
#These keys map each to a list of tuples of the form (x,y,z)

#data_dict=dict.fromkeys(id_list, [(0.0,0.0,0.0)]) #generate a dictionary with specific keys, mapping to an empty list
#print(data_dict)

data_id_list=[]


for class_id in id_list:
	id_centroids=[]
	for row_index in range(len(data_list)):
		if class_id==int(data_array[row_index,3]):
			id_centroids.append((data_array[row_index,0],data_array[row_index,1],data_array[row_index,2]))
	data_id_list.append(id_centroids)

print("IDs of classes observed: ")
print(id_list)

fig1 = plt.figure()
ax = fig1.add_subplot(projection='3d')

xs=data_array[:,0]
ys=data_array[:,1]
zs=data_array[:,2]
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title("Plot showing all YOLOv3 detected object centroids")
plt.show()

fig2=plt.figure()
ax = fig2.add_subplot(projection='3d')
i=0
for point_list in data_id_list: #iterate through dictionary keys
	print(point_list)
	if point_list!=None:
		obj_detections=0
		xs=[]
		ys=[]
		zs=[]
		for v in point_list:
			xs.append(v[0])
			ys.append(v[1])
			zs.append(v[2])
			obj_detections+=1
		print("For class with ID "+str(id)+", this dataset has "+str(obj_detections)+" detections of this class")
		ax.scatter(xs, ys, zs,label=str(id_list[i]))
	i+=1

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
plt.title('YOLOv3 and RGBD detected centroids and their class labels')
plt.show()
