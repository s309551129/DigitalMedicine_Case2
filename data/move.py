import numpy as np
import pydicom
import os
import PIL.Image as Image


# Normalize the pixels' value to (0, 255)
def normalize(arr):
    arr = arr.astype('float')
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (255.0/(maxval-minval))
    return arr

def move(root_from, root_des):
    count = 0
    for dir in os.listdir(root_from):
        dir_next = os.listdir(root_from+dir)
        img_name = os.listdir(root_from+dir+"/"+dir_next[0])[0]
        img_path = root_from + dir + "/" + dir_next[0] + "/" + img_name
        ds = pydicom.dcmread(img_path)
        format_str = 'uint{}'.format(ds.BitsAllocated)
        numpy_dtype = np.dtype(format_str)
        pixel_bytearray = ds.PixelData
        pixel_array = np.frombuffer(pixel_bytearray, dtype=numpy_dtype)
        pixel_array = pixel_array.reshape(ds.Rows, ds.Columns*ds.SamplesPerPixel)
        pixel_array = normalize(pixel_array)
        dcm_img = Image.fromarray(pixel_array).convert('RGB')
        dcm_img.save(root_des+img_name.replace('.dcm', '.jpg'))
        count += 1
        print(count)
    
if __name__ == '__main__':
    root_from = "./data/data/valid/"
    root_des = "./testing/"
    move(root_from, root_des)
    