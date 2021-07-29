import numpy as np
import cv2

x = np.array([[ [111, 112, 113], [121, 122, 123], [131, 132, 133] ],
              [ [211, 212, 213], [221, 222, 223], [231, 232, 233] ],
              [ [311, 312, 313], [321, 322, 323], [331, 332, 333] ]
              ])

#x = np.array([[[0,0,0]]])
img = x #[]
#for i in range(10):
#    img = np.concatenate((img, x), axis=0) if np.size(img) else x
#    print(np.shape(img))
#    #img = np.stack((img, img), axis=1) if np.size(img) else x
#    #print(np.shape(img))

#img = np.expand_dims(img, axis=0)
#print(img)#
#print(np.shape(img))
#img = np.flip(img, axis=0)
#print(img)#
#print(np.shape(img))

print("\n\n\n start")
temp = []
for i in range(2):
    temp.append(img)

print(temp)#
print(np.shape(temp))

img_new = np.stack(temp, axis=0)
print(img_new)#
print(np.shape(img_new))

img_new = np.flip(img_new, axis=2)
print(img_new)
print(np.shape(img_new))


img = cv2.convertScaleAbs(img_new)
#print(img)
#print(np.shape(img))
cv2.imshow("out", img_new)
cv2.waitKey(10000)