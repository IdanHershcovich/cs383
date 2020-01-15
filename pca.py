import os
import numpy as np
from PIL import Image as im

np.set_printoptions(threshold=np.inf) #print all of output without trunc
width = 40
height = 40


my_arr =np.empty(shape= (234,320), dtype=np.uint8) #2darray 234x320
print("initial shape of 2d array: {}".format(my_arr.shape))

face_list = os.listdir('yalefaces') #opens the yalefaces folder
yale_pil = im.open('yalefaces/subject02.centerlight') #single image from yalefaces
resized = yale_pil.resize((40,40)) #resized to 40x40 pix
yale_np = np.asarray(resized,dtype=np.uint8) #resized yale image as array
flat = yale_np.flatten() #flatten to 1d array 1x1600 
print("shape of yale image as flattened array: {}".format(flat.shape))

print(flat)
show_yale = im.fromarray(yale_np)
show_yale.show()








###testing reading all yalefaces into array
# for entry in face_list:
#     og = im.open('yalefaces/'+entry)
#     resized_im = og.resize((40,40))
#     my_arr = np.asarray(resized_im, dtype=np.uint8)


# print(my_arr.shape)
# new_im = im.fromarray(my_arr)
# new_im.show
 


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

    






