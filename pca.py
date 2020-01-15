import os
import numpy as np
from PIL import Image as im

width = 40
height = 40


my_arr =np.empty([234,320], dtype=np.uint8) #2darray 234x320

face_list = os.listdir('yalefaces') #opens the yalefaces folder

for entry in face_list:
    og = im.open('yalefaces/'+entry)
    resized_im = og.resize((40,40))
    my_arr = np.asarray(resized_im, dtype=np.uint8)


# print(my_arr.shape)
new_im = im.fromarray(my_arr)
new_im.show
 


# temp.reshape(1600)
# i = 0
# for new in temp:
#     toim = im.fromarray(new, mode =np.uint8)
#     toim.save('resizedfaces/test' + i)



#### testing resizing and saving into np array
# og = im.open('yalefaces/subject02.centerlight')
# resized = og.resize((40,40))
# array = np.array(resized)
# print(array.shape)
# array.reshape(-1)
# print(array.shape)

###testing saving
# resized.save('test01.gif')
# resized.show()

    






