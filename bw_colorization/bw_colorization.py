#YUV Space
#Takes Y - monochrome luminance channel
#Ouptus U and V, chorminance channels, encoding the color

import cv2
import numpy as np

import cvxopt

from cvxopt.modeling import variable

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix

import os
import errno

from os import path


def rgb2yiq(rgb):
    rgb = rgb / 255.0
    y = np.clip(np.dot(rgb, np.array([0.299, 0.587, 0.144])),             0,   1)
    i = np.clip(np.dot(rgb, np.array([0.595716, -0.274453, -0.321263])), -0.5957, 0.5957)
    q = np.clip(np.dot(rgb, np.array([0.211456, -0.522591, 0.311135])),  -0.5226, 0.5226)
    yiq = rgb
    yiq[..., 0] = y
    yiq[..., 1] = i
    yiq[..., 2] = q
    return yiq


def yiq2rgb(yiq):
    r = np.dot(yiq, np.array([1.0,  0.956295719758948,  0.621024416465261]))
    g = np.dot(yiq, np.array([1.0, -0.272122099318510, -0.647380596825695]))
    b = np.dot(yiq, np.array([1.0, -1.106989016736491,  1.704614998364648]))
    rgb = yiq
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.clip(rgb, 0.0, 1.0) * 255.0

IMG_EXTENSIONS = ["png", "jpeg", "jpg", "gif", "tiff", "tif", "raw", "bmp"]
SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"

filename_input = "giraffe.bmp"
filename_constraints = "giraffe_marked.bmp"

input_image = cv2.imread(filename_input, cv2.IMREAD_GRAYSCALE)

output_bgr = cv2.imread(filename_input)
output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
output_yuv = rgb2yiq(output_rgb)

input_image = (input_image.astype(np.float))/255.

constraint_bgr = cv2.imread(filename_constraints)#, cv2.IMREAD_GRAYSCALE)
constraint_rgb = cv2.cvtColor(constraint_bgr, cv2.COLOR_BGR2RGB)
constraint_yuv = rgb2yiq(constraint_rgb)#cv2.cvtColor(constraint_bgr, cv2.COLOR_BGR2YUV)

#np.set_printoptions(threshold=np.nan)

#print("yuv")
#print(constraint_yuv[:,:,1])

#print("bgr")
#print(constraint_bgr[:,:,1])

#exit()

height, width = input_image.shape

num_pixels = width*height

print(num_pixels)



#Construct sparse diagonal matrix
D = csc_matrix((np.ones(num_pixels),(range(num_pixels),range(num_pixels))),shape=(num_pixels,num_pixels))#np.zeros((num_pixels, num_pixels),dtype = np.float)#np.diag(np.ones(num_pixels))#,dtype=np.float))
#W = np.zeros((num_pixels, num_pixels),dtype = np.float)

W_val=[]
W_i = []
W_j = []

min_sigma = 0.0001

window = 5


#Construct sparse weight matrix
for i_pixel in range(num_pixels):
	
	#D[i,i] = 1.0
	
	h = i_pixel/width
	w = i_pixel%width
	
	count_row = 0
	sum_row = 0.
	
	window_pixels = input_image[max(0,h-window):min(h+window,height-1),max(0,w-window):min(w+window,width-1)]
	#window_pixel_locations = range( 
	
	sigma = max(np.std(window_pixels), min_sigma)
	#print(sigma)
	#if (sigma == 0.0):
	#	print(sigma, min_sigma, np.std(window_pixels))
	
	for j in range(-window, window):
		#for k in range(-window, window):
		for k in range(-window, window):
			if (h+j >= 0 and w+k >= 0 and h+j < height and w+k < width):# and (k != 0 and j != 0)):
				#print(h,h+j,height,w,w+k,width)
				j_pixel = (h+j)*width + (w+k)
				W_val.append(np.exp(-np.square(input_image[h, w] - input_image[h+j, w+k])/(2*sigma**2)))
				W_i.append(i_pixel)
				W_j.append(j_pixel)
				count_row+=1
				sum_row+=np.exp(-np.square(input_image[h, w] - input_image[h+j, w+k])/(2*sigma**2))
	for k in range(count_row):
		W_val[len(W_val)-1-k] /= sum_row
	#print(i_pixel, num_pixels)
		
W = csc_matrix((W_val, (W_i, W_j)),shape=(num_pixels,num_pixels))

print("Created weight matrix")

#M = 0.5*((D-W)+(D-W).T)
#Calculate Laplacian matrix
L = D-W

print("Calculated Laplacian")

#for i in range(num_pixels):
#	sum_row = sum(W[i,:])
#	W[i,:] = W[i,:]/sum_row
	
#print(sum(W[5,:]))
#A_u = np.zeros((num_pixels, num_pixels),dtype = np.float)
#bu = np.zeros((num_pixels),dtype = np.float)
#Au_val = []
#Au_i = []
#Au_j = []

#bu_val = []
#bu_i = []
#bu_j = []

#A_v = np.zeros((num_pixels, num_pixels),dtype = np.float)
#bv = np.zeros((num_pixels),dtype = np.float)
#Av_val = []
#Av_i = []
#Av_j = []

#bv_val = []
#bv_i = []
#bv_j = []

#print(constraint_yuv[:,:,1])

#Convert to lil_matrix for modification of structure
Lu = lil_matrix(L)
Lv = Lu.copy()

#Modify Laplacian matrix to account for fixed colored pixels (i.e. boundary conditions)

print("Converted Laplacian matrix to lil sparse type")

small = 0.001

#rows_u_constraints, cols_u_constraints = np.where(constraint_yuv[:,:,1]>small)
#rows_v_constraints, cols_v_constraints = np.where(constraint_yuv[:,:,2]>small)
rows_u_constraints, cols_u_constraints = np.nonzero(constraint_yuv[:,:,1]-output_yuv[:,:,1])
rows_v_constraints, cols_v_constraints = np.nonzero(constraint_yuv[:,:,2]-output_yuv[:,:,2])
#rows_u_constraints, cols_u_constraints = np.where(np.abs(constraint_yuv[:,:,1]-colored_image_yuv[:,:,1])>small)
#rows_v_constraints, cols_v_constraints = np.where(np.abs(constraint_yuv[:,:,2]-colored_image_yuv[:,:,2])>small)


bu = np.zeros((num_pixels), dtype=np.float)
bv = np.zeros((num_pixels), dtype=np.float)

print(len(rows_u_constraints), ' u constraints')



for index in range(len(rows_u_constraints)):
	i = rows_u_constraints[index]
	j = cols_u_constraints[index]
	pixel = i*width+j
	#print(pixel, num_pixels)
	row_vector = csr_matrix(([1.],([0],[pixel])),shape=(1,num_pixels))
	Lu[pixel,:] = row_vector
	bu[pixel] = constraint_yuv[i,j,1]

print(len(rows_v_constraints), ' v constraints')


for index in range(len(rows_v_constraints)):
	i = rows_v_constraints[index]
	j = cols_v_constraints[index]
	pixel = i*width+j
	row_vector = csr_matrix(([1.],([0],[pixel])),shape=(1,num_pixels))
	Lv[pixel,:] = row_vector
	bv[pixel] = constraint_yuv[i,j,2]	


#Converting back to csc for algebra
Lu_csc = csc_matrix(Lu)
Lv_csc = csc_matrix(Lv)


print("Modified Laplacian with constraints")

u_solution = spsolve(Lu_csc, bu)

print("Solved for U")

v_solution = spsolve(Lv_csc, bv)

print("Solved for V")

u_solution_reshaped = np.reshape(u_solution, (height, width))
v_solution_reshaped = np.reshape(v_solution, (height, width))

#colored_image = rgb2yiq(input_image)
output_yuv[:,:,1] = (u_solution_reshaped)
output_yuv[:,:,2] = (v_solution_reshaped)

output_rgb = np.uint8(yiq2rgb(output_yuv))

output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("output.jpg", output_bgr)


		
#u = np.zeros((num_pixels),dtype = np.floatco)
#v = np.zeros((num_pixels),dtype = np.float)

#Au = cvxopt.spmatrix(Au_val, Au_i, Au_j,size=(num_pixels,num_pixels))
#bu = cvxopt.matrix(bu)#spmatrix(bu_val, bu_i, bu_j)



#Av = cvxopt.spmatrix(Av_val, Av_i, Av_j,size=(num_pixels,num_pixels))
#bv = cvxopt.matrix(bv)#spmatrix(bv_val, bv_i, bv_j)






#q = cvxopt.matrix(np.zeros((num_pixels),dtype = np.float))

#sol_u = cvxopt.solvers.qp(P=M,q=q)#,A=Au,b=bu)

