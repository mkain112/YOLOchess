import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#
# fig = plt.figure()
# nx = 4
# ny = 3
# for i in range(1, nx*ny+1):
#     ax = fig.add_subplot(ny,nx, i)
#     img = aruco.drawMarker(aruco_dict,i, 700)
#     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
#     ax.axis("off")
#
# plt.savefig("markers.pdf")
# plt.show()
frame = cv2.imread('/content/2020-06-09-122022.jpg')
plt.figure()
plt.imshow(frame)
plt.show()
%time
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create( )
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
plt.figure(figsize=(16,16))
plt.imshow(frame_markers)
for i in range(len(ids)):
    c = corners[i][0]
    plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
plt.legend()
plt.show()
def quad_area(data):
    l = data.shape[0]//2
    corners = data[["c1", "c2", "c3", "c4"]].values.reshape(l, 2,4)
    c1 = corners[:, :, 0]
    c2 = corners[:, :, 1]
    c3 = corners[:, :, 2]
    c4 = corners[:, :, 3]
    e1 = c2-c1
    e2 = c3-c2
    e3 = c4-c3
    e4 = c1-c4
    a = -.5 * (np.cross(-e1, e2, axis = 1) + np.cross(-e3, e4, axis = 1))
    return a

corners2 = np.array([c[0] for c in corners])

data = pd.DataFrame({"x": corners2[:,:,0].flatten(), "y": corners2[:,:,1].flatten()},
                   index = pd.MultiIndex.from_product(
                           [ids.flatten(), ["c{0}".format(i )for i in np.arange(4)+1]],
                       names = ["marker", ""] ))

data = data.unstack().swaplevel(0, 1, axis = 1).stack()
data["m1"] = data[["c1", "c2"]].mean(axis = 1)
data["m2"] = data[["c2", "c3"]].mean(axis = 1)
data["m3"] = data[["c3", "c4"]].mean(axis = 1)
data["m4"] = data[["c4", "c1"]].mean(axis = 1)
data["o"] = data[["m1", "m2", "m3", "m4"]].mean(axis = 1)
data
image = cv2.circle(frame, (data.c1[10]['x'],data.c1[10]['y']), radius=3, color=(0, 0, 255), thickness=-1)
image = cv2.circle(frame, (data.c2[10]['x'],data.c2[10]['y']), radius=3, color=(0,255, 255), thickness=-1)
image = cv2.circle(frame, (data.c4[10]['x'],data.c4[10]['y']), radius=3, color=(255,0, 255), thickness=-1)
#lets call this the y axis
y2=data.c2[10]['y']
y1=data.c1[10]['y']
x2=data.c2[10]['x']
x1=data.c1[10]['x']
m=(y2-y1)/(x2-x1)
b=y1-(m*x1)
image= cv2.line(frame, (data.c2[10]['x'],data.c2[10]['y']),((-b/m),0),(255,0,0),3)

#lets call this the x axis
y4=data.c4[10]['y']
x4=data.c4[10]['x']
m2=(y1-y4)/(x1-x4)
b2=y1-(m2*x1)
end=image.shape[1]
print(image.shape)
image= cv2.line(frame, (data.c4[10]['x'],data.c4[10]['y']),(end,int(end*m2+b2)),(255,0,0),3)
cv2_imshow(image)
