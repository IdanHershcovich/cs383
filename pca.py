import os
import numpy as np
from PIL import Image as im

np.set_printoptions(threshold=np.inf) #print all of output without trunc
width = 40
height = 40

empty_matrix = np.empty([0,1600], dtype=np.uint8)

# yale_pil = im.open('yalefaces/subject02.centerlight') #single image from yalefaces
# resized = yale_pil.resize((40,40)) #resized to 40x40 pix
# yale_np = np.asarray(resized,dtype=np.uint8) #resized yale image as array
# flat = yale_np.flatten() #flatten to 1d array 1x1600 




###testing reading all yalefaces into array

def resize_im(image, h, w):
    resized = image.resize((h,w))
    return resized


###from a given directory, loads all images, reduces size to 40x40, flattens it to a 1d array and then concatenates it to a matrix specified by the user
def yaleMatrix(directory, matrix):
    face_list = os.listdir(directory) #opens the yalefaces folder
    for entry in face_list:
        og = im.open(directory+'/'+entry)
        res_im = resize_im(og, height, width)
        im_as_arr = np.asarray(res_im, dtype=np.uint8)
        flat = im_as_arr.flatten()
        flat = np.column_stack(flat)
        matrix = np.concatenate((matrix, flat), axis=0)
       
   
    print(matrix.shape)
    standarized = np.std(matrix)
    print(standarized)
    return matrix



yaleMatrix('yalefaces', empty_matrix)

# print(str.format("shape of matrix: {}", flat.shape))


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




