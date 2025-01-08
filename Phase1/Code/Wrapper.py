#!/usr/bin/env python

"""
CMSC733 Spring 2024: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Abubakar Siddiq Palli (absiddiq@umd.edu)
MEng Robotics, 2nd Semester,
University of Maryland, College Park

"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

dog_global_scale = []
dog_global_orient = []

txtn_scales = [3, 5, 9]
txtn_orientations = np.linspace(0, 360, 10)
lml_scales = [np.sqrt(2), 2, 2*np.sqrt(2)]
lml_orientations = np.linspace(0, 360, 6)
lms_scales = [1,np.sqrt(2), 2]
lms_orientations = np.linspace(0, 360, 6)



def gaussian_kernel_DOG(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)
    

def generate_dog_filter_bank(scales, orientations):
    
    global dog_global_scale, dog_global_orient
    dog_global_scale = scales
    dog_global_orient = orientations
    filters = [] 
    for scale in scales: 
        for orientation in orientations:
           
            border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)

            sigma = scale
            size = 50

            if size % 2 == 0:
                size += 1
            # Generate Gaussian filter
            gaussian_filter = gaussian_kernel_DOG(size, sigma)
            rotation_matrix = cv2.getRotationMatrix2D((gaussian_filter.shape[1]/2, gaussian_filter.shape[0]/2), orientation, 1)
            #convolcing gaussian with sobel
            dog_filter = cv2.Sobel(gaussian_filter,cv2.CV_64F,1,0,ksize=3, borderType=border)
            rotated_DOG_filter = cv2.warpAffine(dog_filter, rotation_matrix, (dog_filter.shape[1], dog_filter.shape[0]))
            filters.append(rotated_DOG_filter)
    
    return filters

def display_and_save_dog_filters(filters, save_path='Phase1/Results/DoG.png'):
    rows = len(dog_global_scale)
    cols = len(dog_global_orient)
    
    plt.figure(figsize=(12, 4))
    
    for i, filter_i in enumerate(filters):
        plt.subplot(rows, cols, i+1)
        plt.imshow(filter_i, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def gaussian_kernel_2D(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_kernel_2d_elongated(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (3*y - (3*size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def laplace_2d_gaussian(size, sigma):
  
    # Generate a grid of squared distances
    x, y = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))
    squared_distances = x**2 + y**2

    # Calculate the Gaussian distribution
    gaussian = (1 / (2 * np.pi * sigma**2)) * np.exp(-squared_distances / (2 * sigma**2))

    # Calculate the Laplacian of the Gaussian
    laplaced_gaussian = gaussian * ((squared_distances - sigma**2) / (sigma**4))

    return laplaced_gaussian

def generate_lm_filter_bank(scales, orientations):

    scales_laplace = [np.sqrt(2), 2, 2*np.sqrt(2), 4]
    filters = []
    size = 30
    #sobel kernals
    Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])

    first_derivatives = []
    second_derivatives = []
    for scale in scales:
        for orientation in orientations:
            Gauss2d_elong = gaussian_kernel_2d_elongated(size,scale)

            rotation_matrix = cv2.getRotationMatrix2D((Gauss2d_elong.shape[1]/2, Gauss2d_elong.shape[0]/2), orientation, 1)
            #convolving gaussian with sobel
            first_derivative = cv2.filter2D(Gauss2d_elong, -1, Sx) + cv2.filter2D(Gauss2d_elong, -1, Sy)
            convolved_1st_filters = cv2.warpAffine(first_derivative, rotation_matrix, (first_derivative.shape[1], first_derivative.shape[0]))
            first_derivatives.append(convolved_1st_filters)
            second_derivative = cv2.filter2D(first_derivative, -1, Sx) + cv2.filter2D(first_derivative, -1, Sy)
            convolved_2nd_filters = cv2.warpAffine(second_derivative, rotation_matrix, (second_derivative.shape[1], second_derivative.shape[0]))
            second_derivatives.append(convolved_2nd_filters)

    filters+=first_derivatives
    filters+=second_derivatives

    for i in range(len(scales_laplace)):
        filters.append(gaussian_kernel_2D(size, scales_laplace[i]))   
    for i in range(len(scales_laplace)):
        filters.append(laplace_2d_gaussian(size, scales_laplace[i]))
    for i in range(len(scales_laplace)):
        filters.append(laplace_2d_gaussian(size, 3*scales_laplace[i]))
    return filters


def display_and_save_LMS_filters(filters, save_path='Phase1/Results/LM.png'):
    rows = 4
    cols = 12

    plt.figure(figsize=(12, 4))
    
    for i, filter_i in enumerate(filters):
        plt.subplot(rows, cols, i+1)
        plt.imshow(filter_i, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def generate_Gabor_filter(sigma, theta, Lambda, psi, gamma, size):
    filter_result = np.zeros((size, size))

    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * np.pi * x_theta / Lambda  + psi)
    sinusoidal_part = np.cos(2 * np.pi / Lambda * x_theta + psi)
    filter_result = gb * sinusoidal_part 

    filter_result /= np.sum(np.abs(filter_result))

    return filter_result


def generate_Gabor_filter_bank(orientations, scales, size):
    filter_bank = []

    for scale in scales:
        for orientation in orientations:
            # Calculate wavelength and phase offset based on scale and orientation
            wavelength = 0.8 * scale
            phase_offset = orientation
            # Generate Gabor filter with specified parameters
            gabor_filter = generate_Gabor_filter(sigma=5*scale, theta=phase_offset, Lambda=wavelength, psi=np.pi/2, gamma=3, size=size)

            # Normalize the filter to have a unit sum and add it to the filter bank
            gabor_filter /= np.sum(np.abs(gabor_filter))
            filter_bank.append(gabor_filter)
    return filter_bank


def display_and_save_Gabor_filter_bank(filters, save_path='Phase1/Results/Gabor.png'):
    rows = 5
    cols = 8 
    
    plt.figure(figsize=(12, 4))
    
    for i, (filter_i) in enumerate(filters):
        plt.subplot(rows, cols, i+1)
        plt.imshow(filter_i, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    plt.show()

def generate_half_disc_map(radius, resolution=0.02):
    # Create a meshgrid for x and y coordinates
    x = np.arange(-radius, radius + resolution, resolution)
    y = np.arange(-radius, radius + resolution, resolution)
    xx, yy = np.meshgrid(x, y)

    # Calculate the distance from the center for each point
    distance = np.sqrt(xx**2 + yy**2)

    # Create a mask for the half-disc
    half_disc_mask = (distance <= radius) & (xx >= 0)

    return half_disc_mask.astype(np.uint8)  # Convert boolean to uint8

def half_disc_bank(scales, orientations, half_disc):
    half_disc_bank = []
    for scale in scales:
        for orientation in orientations:
            rotation_matrix = cv2.getRotationMatrix2D((half_disc.shape[1]/2, half_disc.shape[0]/2), orientation, 1)
            rotated_half_disc = cv2.warpAffine(half_disc, rotation_matrix, (half_disc.shape[1], half_disc.shape[0]))

            # Resize without maintaining aspect ratio
            resized_half_disc = cv2.resize(rotated_half_disc, None, fx=scale/100, fy=scale/100)

            half_disc_bank.append(resized_half_disc)

            flipped_half_disc_x = cv2.flip(resized_half_disc,0)
            flipped_half_disc_xy = cv2.flip(flipped_half_disc_x,1)

            half_disc_bank.append(flipped_half_disc_xy)

    return half_disc_bank

def display_and_save_half_disc(bank, scales, orientations, save_path='Phase1/Results/half_disc.png'):
    rows = 2*len(scales)
    cols = len(orientations)
    
    plt.figure(figsize=(12,8))
    
    for i, disk_i in enumerate(bank):
        plt.subplot(rows, cols, i+1)
        plt.imshow(disk_i, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def filter_convolve_with_image(filter, image):
    filter_responses = []
    for channel in range(image.shape[-1]):
        channel_responses = cv2.filter2D(image[:, :, channel], -1, filter)
        filter_responses.append(channel_responses)
    filter_responses = np.array(filter_responses)
    responses_matrix = np.moveaxis(filter_responses, 0, -1)

    return responses_matrix


def generate_texton_map(image):
    # Generate filter banks
    dog_filter_bank = generate_dog_filter_bank(txtn_scales, txtn_orientations)
    lml_filter_bank = generate_lm_filter_bank(lml_scales,lml_orientations)
    lms_filter_bank = generate_lm_filter_bank(lms_scales,lms_orientations)
    gabor_filter_bank = generate_Gabor_filter_bank(txtn_orientations, txtn_scales, 30)
    #Complie all the filter banks
    complete_filter_bank  = dog_filter_bank + lml_filter_bank + lms_filter_bank + gabor_filter_bank
    # Initialize an empty array for the filter responses
    all_responses_flat = []

    # Apply each filter and concatenate responses
    for filter in complete_filter_bank:
        responses_matrix = filter_convolve_with_image(filter, image)
        responses_flat = responses_matrix.reshape((-1, responses_matrix.shape[-1]))
        all_responses_flat.append(responses_flat)

    all_responses_flat = np.concatenate(all_responses_flat, axis=-1)

    K = 5

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=K, random_state=0)
    texton_ids = kmeans.fit_predict(all_responses_flat)
    
    # Reshape the texton_ids back to the shape of the image
    texton_map = texton_ids.reshape(image.shape[:2])
    return texton_map

def generate_brightness_map(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = gray_image.shape

    reshaped_image = gray_image.reshape(h*w,1)

    # Perform k-means clustering
    K = 8
    kmeans = KMeans(n_clusters=K, random_state=0)
    texton_ids = kmeans.fit_predict(reshaped_image)

    # Reshape the texton_ids back to the shape of the image
    brightness_map = texton_ids.reshape(h,w)
    return brightness_map

def generate_color_map(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,w,_ = hsv_image.shape
    
    channel1_image = hsv_image[:,:,0]
    channel2_image = hsv_image[:,:,1]
    channel3_image = hsv_image[:,:,2]

    channel1 = channel1_image.reshape(h*w,1)
    channel2 = channel2_image.reshape(h*w,1)
    channel3 = channel3_image.reshape(h*w,1)

    K = 5
    kmeans = KMeans(n_clusters=K, random_state=0)

    channel1_ids = kmeans.fit_predict(channel1)
    channel1_color_map = channel1_ids.reshape(h, w)

    channel2_ids = kmeans.fit_predict(channel2)
    channel2_color_map = channel2_ids.reshape(h, w)

    channel3_ids = kmeans.fit_predict(channel3)
    channel3_color_map = channel3_ids.reshape(h, w)

    merged_image = channel1_color_map+channel2_color_map+channel3_color_map

    return merged_image


def compute_chi_square(img, left_mask, right_mask):
    chi_sqr_dist = np.zeros_like(img, dtype=np.float32)
    num_bins = 56 

    for i in range(num_bins):
        tmp = np.where(img == i, 1, 0).astype(np.float32)
        g_i = cv2.filter2D(tmp, -1, left_mask)
        h_i = cv2.filter2D(tmp, -1, right_mask)
        # Adding a small value to avoid division by zero
        chi_sqr_dist += ((g_i - h_i) ** 2) / (g_i + h_i + 1e-10)

    return chi_sqr_dist

def generate_gradient(image, bank):
	
	final_gradient = np.zeros(image.shape)
	for i in range(len(bank)):
		if i%2==0:
			left_mask = bank[i]
			right_mask = bank[i+1]
    
			gradient_map = compute_chi_square(image,left_mask,right_mask)
			final_gradient +=gradient_map
    
	return final_gradient

def main():
    
    scales = [3, 5, 9]
    orientations = np.linspace(0, 360, 10)
    # Generate, Display and save DOG filters
    dog_filters = generate_dog_filter_bank(scales, orientations)
    display_and_save_dog_filters(dog_filters)


    lm_scales = [np.sqrt(2), 2, 2*np.sqrt(2)]
    lm_orientations = np.linspace(0, 360, 6)
    # Generate, Display and save LM filters
    lms_filters = generate_lm_filter_bank(lm_scales,lm_orientations) 
    display_and_save_LMS_filters(lms_filters)

    gabor_scales = [8,10,13,20,25]
    gabor_orientations = np.linspace(0, np.pi,8)
    filter_size = 30
	# Generate, Display and save Gabor filters
    Gabor_filter_bank = generate_Gabor_filter_bank(gabor_orientations, gabor_scales, filter_size)
    display_and_save_Gabor_filter_bank(Gabor_filter_bank)


    #generate half-disc bank 
    radius = 5.0
    half_disc = generate_half_disc_map(radius)
    hd_scales = [2,5,10]
    hd_orientations = np.linspace(0, 360, 8)
    half_disc_bank_result = half_disc_bank(hd_scales, hd_orientations, half_disc)
    # Display the half-disc filter bank
    display_and_save_half_disc(half_disc_bank_result, scales, orientations)

    #Result folders for the Generated Images
    texton_result_folder = r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\Results\Texton"
    texton_gradient_result_folder = r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\Results\Tg"
    brightness_result_folder = r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\Results\Brightness"
    brightness_gradient_result_folder = r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\Results\Bg"
    color_result_folder = r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\Results\Color"
    color_gradient_result_folder = r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\Results\Cg"
    pb_lite_results_folder = r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\Results\Pb_lite"


    #Loop to pass each input image - as images are number from 1.jpg to 10.jpg

    for i in range(1,11):

        #Generate Texton Map
        texton_image = cv2.imread(r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\BSDS500\Images\{}.jpg".format(i))
        texton_map = generate_texton_map(texton_image)
        texton_result_path = os.path.join(texton_result_folder, 'TextonMap_{}.png'.format(i))
        plt.imshow(texton_map, cmap='viridis')
        plt.axis('off')
        plt.savefig(texton_result_path)
        plt.close()

        #Generate Texton Map Gradient
        Tg = generate_gradient(texton_map, half_disc_bank_result)
        Tg_result_path = os.path.join(texton_gradient_result_folder, 'Texton_Gadient_{}.png'.format(i))
        plt.imshow(Tg,cmap="gray")
        plt.axis('off')
        plt.title("Texture Gradient")
        plt.axis('off')
        plt.savefig(Tg_result_path)
        plt.close()

        #Generate Brightness Map 
        brightness_image = cv2.imread(r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\BSDS500\Images\{}.jpg".format(i))
        brightness_map = generate_brightness_map(brightness_image)
        brightness_result_path = os.path.join(brightness_result_folder, 'BrightnessMap_{}.png'.format(i))
        plt.imshow(brightness_map, cmap='gray')
        plt.axis('off')
        plt.savefig(brightness_result_path)
        plt.close()

        #Generate Brightness Map Gradient
        Bg = generate_gradient(brightness_map, half_disc_bank_result)
        Bg_result_path = os.path.join(brightness_gradient_result_folder, 'Brightness_Gadient_{}.png'.format(i))
        plt.imshow(Bg,cmap="gray")
        plt.axis('off')
        plt.title("Brightness Gradient")
        plt.axis('off')
        plt.savefig(Bg_result_path)
        plt.close()

        #Generate Color Map 
        color_image = cv2.imread(r"C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\BSDS500\Images\{}.jpg".format(i))
        color_map = generate_color_map(color_image)
        color_result_path = os.path.join(color_result_folder, 'ColorMap_{}.png'.format(i))
        plt.imshow(color_map)
        plt.axis('off')
        plt.savefig(color_result_path)
        plt.close()
        
        #Generate Color Map Gradient
        Cg = generate_gradient(color_map, half_disc_bank_result)
        Cg_result_path = os.path.join(color_gradient_result_folder, 'Color_Gadient_{}.png'.format(i))
        plt.imshow(Cg,cmap="gray")
        plt.axis('off')
        plt.title("Color Gradient")
        plt.savefig(Cg_result_path)
        plt.close()

        #Read and Convert the Canny and Sobel baselines to Gray scale
        sobelPb = cv2.imread(r'C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\BSDS500\SobelBaseline\{}.png'.format(i))
        sobelPb_gray = cv2.cvtColor(sobelPb,cv2.COLOR_BGR2GRAY)
        cannyPb = cv2.imread(r'C:\Users\abuba\Desktop\CMSC733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\BSDS500\CannyBaseline\{}.png'.format(i))
        cannyPb_gray = cv2.cvtColor(cannyPb,cv2.COLOR_BGR2GRAY)

        #Reshape the Gradient maps so that they are compatible with Canny and Sobel baselines
        Tg_reshaped = cv2.resize(Tg, (cannyPb.shape[1],cannyPb.shape[0]) )
        Bg_reshaped = cv2.resize(Bg, (cannyPb.shape[1],cannyPb.shape[0]) )
        Cg_reshaped = cv2.resize(Cg, (cannyPb.shape[1],cannyPb.shape[0]) ) 

        #weights for Pb-lite algorithm
        w1 = 0.9
        w2 = 0.1

        #Pb-lite implementation
        p_b = ((Tg_reshaped + Bg_reshaped + Cg_reshaped) / 3) * (w1 * cannyPb_gray + w2 * sobelPb_gray)
        pb_result_path = os.path.join(pb_lite_results_folder, 'Pb_lite_boundary_{}.png'.format(i))
        
        #Plotting the boundary detected using Pb-lite
        plt.imshow(p_b,cmap="gray")
        plt.axis('off')
        plt.savefig(pb_result_path)
        plt.show()

if ( __name__ == '__main__'): 
    main()