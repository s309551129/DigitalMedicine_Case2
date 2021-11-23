import pydicom
import os
import numpy
import PIL.Image as Image
import pydicom

# Normalize the pixels' value to (0, 255)
def normalize(arr):
    arr = arr.astype('float')
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (255.0/(maxval-minval))
    return arr

# Convert DCM to JPG
def dcm2jpg(dcm_path, dcm_file, img_path):
    ds = pydicom.read_file(os.path.join(dcm_path, dcm_file))
    format_str = 'uint{}'.format(ds.BitsAllocated)
    numpy_dtype = numpy.dtype(format_str)
    pixel_bytearray = ds.PixelData
    pixel_array = numpy.frombuffer(pixel_bytearray, dtype=numpy_dtype)
    pixel_array = pixel_array.reshape(ds.Rows, ds.Columns*ds.SamplesPerPixel)
    newi = pixel_array
    pixel_array = normalize(pixel_array)
    dcm_img = Image.fromarray(pixel_array).convert('RGB')
    img_name = dcm_file.split('.')[0] + '.jpg'
    dcm_img.save(os.path.join(img_path, img_name))

# Train Images
mode = 'train'
dcm_path = os.path.join('data', mode)

img_path = './data/{}_images'.format(mode)
if not os.path.exists(img_path):
    os.mkdir(img_path)

filelist = os.listdir(dcm_path)
print(len(filelist))
for dcm_file in filelist:
    dcm2jpg(dcm_path, dcm_file, img_path)

# Valid Images
mode = 'valid'
dcm_path = os.path.join('data', mode)

img_path = './data/{}_images'.format(mode)
if not os.path.exists(img_path):
    os.mkdir(img_path)

filelist = os.listdir(dcm_path)
print(len(filelist))
for dcm_file in filelist:
    dcm2jpg(dcm_path, dcm_file, img_path)