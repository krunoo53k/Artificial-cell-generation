from __future__ import annotations
import argparse
import cmath
from math import atan2
from random import random
from random import randint
from random import choice
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.special import comb
from scipy.interpolate import interp1d
from skimage.transform import rescale
from skimage.util import random_noise
from matplotlib.colors import LinearSegmentedColormap
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)
from math import pi
import asyncio

from abc import ABC, abstractmethod
from typing import List

colors = [(0.62,0.54,0.51), (0.2,0.07,0.52)] # first color is black, last is red
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)

colors = [(0.94,0.83,0.73), (0.84,0.68,0.66)] # first color is black, last is red
cellcmap = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)

colors = [(0.64,0.55,0.61),(0.16,0.1,0.53)] # first color is black, last is red
monocytecmap = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)



def convexHull(pts):    #Graham's scan.
    xleftmost, yleftmost = min(pts)
    by_theta = [(atan2(x-xleftmost, y-yleftmost), x, y) for x, y in pts]
    by_theta.sort()
    as_complex = [complex(x, y) for _, x, y in by_theta]
    chull = as_complex[:2]
    for pt in as_complex[2:]:
        #Perp product.
        while ((pt - chull[-1]).conjugate() * (chull[-1] - chull[-2])).imag < 0:
            chull.pop()
        chull.append(pt)
    return [(pt.real, pt.imag) for pt in chull]


def dft(xs):
    return [sum(x * cmath.exp(2j*pi*i*k/len(xs)) 
                for i, x in enumerate(xs))
            for k in range(len(xs))]

def interpolateSmoothly(xs, N):
    """For each point, add N points."""
    fs = dft(xs)
    half = (len(xs) + 1) // 2
    fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
    return [x.real / len(xs) for x in dft(fs2)[::-1]]

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def noisy(noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def generate_blob_image_beizer(size=100):
    pts = [(random() + 0.8) * cmath.exp(2j*cmath.pi*i/7) for i in range(7)]
    pts = convexHull([(pt.real, pt.imag) for pt in pts])
    xs, ys = bezier_curve(pts)

    # prazna slika size x size
    img = np.zeros((size, size))
    # prebacivanje u koordinate
    points = np.array(list(zip(xs, ys)))
    # normaliziranje da bude izmedju 0 i 1
    points += 2
    points /= 2 + 2
    # skaliranje na velicinu slike
    points *= size
    # openCV format
    points = points.astype(np.int32)
    points = points.reshape((-1, 1, 2))
    cv.fillPoly(img, [points], color=255)
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    return img


def transform_blob(img):
    img2 = distance_transform_edt(img)
    img = img - img2
    img[img != 0] = (img[img != 0] - np.min(img[img != 0])) / (np.max(img[img != 0]) - np.min(img[img != 0]))
    img = gaussian_filter(img, sigma=2)
    return img

def color_blob(img):
    #grayscale_blob = img.copy()
    img = transform_blob(img)
    img = cellcmap(img)
    img = img[:,:,:3] #cut alpha channel
    # Modify img where blob has a value of 0
    img[np.all(img == (0.94,0.83,0.73), axis=-1)] = [0, 0, 0]  # Setting background to zero
    return img

def generate_background_image():
    # prazna slika size x size
    max_blob_size = 140
    min_blob_size = 60
    #img = np.zeros((360 + max_blob_size, 363 + max_blob_size, 3))
    img = np.full((360 + max_blob_size, 363 + max_blob_size, 3),(0.94,0.83,0.73))
    num_of_blobs = randint(10, 20)
    for i in range(0, num_of_blobs):
        size = randint(min_blob_size, max_blob_size)
        centre_x = randint(0, 360 + max_blob_size - size)
        centre_y = randint(0, 363 + max_blob_size - size)
        blob = generate_blob_image(size)
        blob = color_blob(blob)

        # Binary mask to ignore zeros in the blob image
        mask = np.all(blob != [0,0,0], axis=-1)
        
        img[centre_x:centre_x+size, centre_y:centre_y+size][mask] = blob[mask]

    #img = gaussian_filter(img, sigma=0.1)
    #return img
    #plt.imshow(img)
    #plt.show()
    return img[max_blob_size::,max_blob_size::,::]

def add_colour(background_image, color):
    # Expand the grayscale image to have 3 channels
    color_image = np.stack((background_image,) * 3, axis=-1)

    # Multiply each pixel value of the color image with the color value
    result_image = color_image * color

    # Clip values to [0, 255]
    result_image = np.clip(result_image, 0, 255)

    return result_image

def add_background(image, color):
    mask = np.all(image >= [209, 209, 209], axis=-1)
    image[mask] = color
    return image


def invert_colors(image):
    # Invert colors by subtracting each pixel value from 255
    inverted_image = np.full(image.shape,255)
    inverted_image = np.subtract(inverted_image, image)

    return inverted_image

def generate_cell(size = 512):
    img = generate_blob_image(size)
    blob = img.copy()
    img = random_noise(img, 'pepper', amount=0.2)
    img = gaussian_filter(img, sigma=1)
    colors = [ (0.45,0.24,0.627), (0.83,0.79,0.73)] # first color is black, last is red
    colourmap = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)
    img = colourmap(img)
    img = img[:,:,:3] #cut alpha channel
    # Modify img where blob has a value of 0
    img[blob == 0] = [0, 0, 0]  # Setting background to zero
    return img


def generate_nucleus(size = 256):
    img = generate_neutrophil_nucleus(size)
    blob = img.copy()
    #np.seed(0)
    noise = generate_fractal_noise_2d((256, 256), (8, 8), 5)
    noise = (noise-np.min(noise))/(np.max(noise)-np.min(noise)) #normalize the noise from 0 to 1
    # Find indices of non-zero values in img
    nonzero_indices = np.nonzero(img)
    # Add noise only to non-zero values
    img[nonzero_indices] = noise[nonzero_indices]
    img = cm(img)
    img = img[:,:,:3] #cut alpha channel
    # Modify img where blob has a value of 0
    img[blob == 0] = [0, 0, 0]  # Setting background to zero
    img = gaussian_filter(img, sigma=0.3)
    return img

def generate_full_background():
    background_image = generate_background_image()
    background_image = gaussian_filter(background_image, sigma=1)

    # Define a static color (e.g., yellow)
    color = np.array([174/255,190/255,188/255])
    #color = np.array([204/255, 160/255, 39/255])  # Yellow

    # Add color to the grayscale background image
    colored_image = add_colour(background_image, color)
    print(colored_image.shape)
    colored_image = invert_colors(colored_image)
    colored_image = add_background(colored_image, np.array([255,233,203]))
    #colored_image = noisy("poisson", colored_image)
    #colored_image = add_background(colored_image, np.array([0.5,0.5,0.5]))
    # Display the result
    plt.figure
    plt.imshow(colored_image.astype(np.uint8))
    plt.show()
    return colored_image

def remove_null_rows_and_columns_rgb_image(img):
    zero_rows = np.all(np.sum(img, axis=2) == 0, axis=1)
    zero_cols = np.all(np.sum(img, axis=2) == 0, axis=0)

    # Remove zero rows and columns
    img = img[~zero_rows][:, ~zero_cols]
    #plt.imshow(img)
    #plt.show()
    return img


def generate_neutrophil_image():
    cell = generate_neutrophil()
    cell = remove_null_rows_and_columns_rgb_image(cell)
    x = randint(10, 360-cell.shape[0]-10)
    y = randint(10, 363-cell.shape[1]-10)
    print("x, y, w, h", x+cell.shape[0]/2, y+cell.shape[1]/2, cell.shape[0], cell.shape[1])
    background = generate_background_image()
    mask = np.all(cell >= [0.05,0.05,0.05], axis=-1)
    background[x:x+cell.shape[0],y:y+cell.shape[1]][mask]=cell[mask]
    #background = gaussian_filter(background, sigma=0.5)
    #print(background.shape)
    x = (x + cell.shape[0]/2) / background.shape[1]
    y = (y + cell.shape[1]/2) / background.shape[0]
    h = cell.shape[0]/background.shape[0]
    w = cell.shape[1]/background.shape[1]

    yolobbox=(y, x, w, h)
    return background, yolobbox

def generate_monocyte_image():
    cell = generate_monocyte()
    cell = remove_null_rows_and_columns_rgb_image(cell)
    x = randint(10, 360-cell.shape[0]-10)
    y = randint(10, 363-cell.shape[1]-10)
    #print("x, y, w, h", x+cell.shape[0]/2, y+cell.shape[1]/2, cell.shape[0], cell.shape[1])
    background = generate_background_image()
    mask = np.all(cell >= [0.05,0.05,0.05], axis=-1)
    background[x:x+cell.shape[0],y:y+cell.shape[1]][mask]=cell[mask]
    #background = gaussian_filter(background, sigma=0.5)
    #print(background.shape)
    x = (x + cell.shape[0]/2) / background.shape[1]
    y = (y + cell.shape[1]/2) / background.shape[0]
    h = cell.shape[0]/background.shape[0]
    w = cell.shape[1]/background.shape[1]

    yolobbox=(y, x, w, h)
    return background, yolobbox

def generate_monocyte(size = 512):
    img = generate_blob_image(size)
    blob = img.copy()
    underlayer = rescale(img, 1.25, anti_aliasing=False)
    underlayerblob = underlayer.copy()
    underlayer = random_noise(underlayer, 'pepper', amount=0.5)
    underlayer = monocytecmap(underlayer)
    underlayer = underlayer[:,:,:3]
    underlayer[underlayerblob==0] = [0,0,0]
    underlayer = rescale(underlayer, 0.25, channel_axis=2, anti_aliasing=False)
    #plt.imshow(underlayer)
    #plt.show()
    #plt.imshow(img)
    #plt.show()
    img = random_noise(img, 'pepper', amount=0.2)
    img = monocytecmap(img)
    img = img[:,:,:3]
    img[blob==0] = [0,0,0]
    img = rescale(img, 0.25, channel_axis=2, anti_aliasing=False)
    #plt.imshow(img)
    #plt.show()
    difference = underlayer.shape[0]-img.shape[0]
    difference = int(difference/2)
    mask = np.all(img != [0,0,0], axis=-1)
    underlayer[difference:img.shape[0]+difference,difference:img.shape[0]+difference][mask] = img[mask]
    underlayer = gaussian_filter(underlayer, sigma=0.5)
    
    return underlayer



def generate_neutrophil_images(count=10):
    counter = 0
    for i in range(0,count):
        image, yolobox = generate_neutrophil_image()
        plt.imsave("output/neutrophil/image"+str(i)+".jpg",image)
        #cv.imshow("neutrofil",image)
        file = open("output/neutrophil/image"+str(i)+".txt",'w')
        file.write("0 "+str(yolobox[0])+" "+str(yolobox[1])+ " " + str(yolobox[2])+ " " + str(yolobox[3]))
        file.close()
        counter+=1
        print("Progress: ",counter," out of ", count, " images, ",(counter/count)*100,"%")

def generate_monocyte_images(count=10, start=0):
    counter = 0
    for i in range(start,count+start):
        image, yolobox = generate_monocyte_image()
        plt.imsave("output/monocyte/image"+str(i)+".jpg",image, vmin=0, vmax=1)
        file = open("output/monocyte/image"+str(i)+".txt",'w')
        file.write("1 "+str(yolobox[0])+" "+str(yolobox[1])+ " " + str(yolobox[2])+ " " + str(yolobox[3]))
        file.close()
        counter+=1
        print("Progress: ",counter," out of ", count, " images, ",(counter/count)*100,"%")

def generate_neutrophil_nucleus():
    padding = 0
    img = np.full((512 + padding, 512 + padding), 0)
    blob_size = int(256)
    num_blobs = 4
    
    for _ in range(num_blobs):
        i = np.random.randint(0, 2)
        j = np.random.randint(0, 2)
        blob = generate_blob_image(blob_size)
        
        img[blob_size * i:blob_size * (i + 1), blob_size * j:blob_size * (j + 1)] = blob
    
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    mid_row = int(img.shape[0] / 2)
    mid_column = int(img.shape[1] / 2)

    non_zero_indices = np.where(img != 0)
    non_zero_values = img[non_zero_indices]

    # Get the bottom half of the image
    bottom_half_indices = (non_zero_indices[0] >= mid_row)
    bottom_half_values = non_zero_values[bottom_half_indices]

    # Shift the bottom half non-zero values up by 30 pixels
    shifted_indices = (non_zero_indices[0][bottom_half_indices] - 150, non_zero_indices[1][bottom_half_indices])
    img[non_zero_indices[0][bottom_half_indices], non_zero_indices[1][bottom_half_indices]] = 0  # Set non-zero values to zero
    img[shifted_indices] = bottom_half_values  # Place non-zero values in the shifted positions

    non_zero_indices = np.where(img != 0)
    non_zero_values = img[non_zero_indices]

    # Get the right half of the image (you can adjust 'mid_column' accordingly)
    right_half_indices = (non_zero_indices[1] >= mid_column)
    right_half_values = non_zero_values[right_half_indices]

    # Shift the right half non-zero values left by 30 pixels
    shifted_indices = (non_zero_indices[0][right_half_indices], non_zero_indices[1][right_half_indices] - 150)
    img[non_zero_indices[0][right_half_indices], non_zero_indices[1][right_half_indices]] = 0  # Set non-zero values to zero
    img[shifted_indices] = right_half_values  # Place non-zero values in the shifted positions

 #   plt.imshow(img, cmap="Greys")
 #   plt.show()
    zero_rows = np.all(img == 0, axis=1)
    zero_cols = np.all(img == 0, axis=0)

    # Remove zero rows and columns
    img = img[~zero_rows][:, ~zero_cols]

    #img[img==255]=1

    return img

def color_neutrophil_nucleus(img):
    blob = img.copy()
    noise = generate_fractal_noise_2d((512, 512), (8, 8), 5)
    #plt.imshow(noise, cmap="Greys")
    #plt.show()
    # Crop the noise to match the size of img
    noise = noise[:img.shape[0], :img.shape[1]]
    noise = (noise-np.min(noise))/(np.max(noise)-np.min(noise)) #normalize the noise from 0 to 1

    img = noise
    img = cm(img)
    img = img[:,:,:3] #cut alpha channel
    # Modify img where blob has a value of 0
    img[blob == 0] = [0, 0, 0]  # Setting background to zero
    #img = gaussian_filter(img, sigma=0.3)
    #plt.imshow(img)
    #plt.show()
    return img

def generate_neutrophil():
    neutrophil = generate_neutrophil_nucleus()
    neutrophil = color_neutrophil_nucleus(neutrophil)
    cell = generate_cell(750)
    non_zero_indices = np.where(neutrophil != 0)

    # Calculate the coordinates to place neutrophil in the middle of cell
    cell_height, cell_width = cell.shape[:2]
    neutrophil_height, neutrophil_width = neutrophil.shape[:2]
    
    y_start = (cell_height - neutrophil_height) // 2
    y_end = y_start + neutrophil_height
    x_start = (cell_width - neutrophil_width) // 2
    x_end = x_start + neutrophil_width

    # Copy neutrophil into the cell at the calculated coordinates
    cell[y_start:y_end, x_start:x_end][non_zero_indices] = neutrophil[non_zero_indices]

    cell = rescale(cell, 0.25, channel_axis=2, anti_aliasing=False)

    return cell


parser = argparse.ArgumentParser(
                    prog='Cell Generator',
                    description='Generate neutrophil or monocyte images',
                    epilog='Hello!')

"""
parser.add_argument('-c','--count', type=int)
parser.add_argument('-t', '--type')

args = parser.parse_args()

if(args.type=="monocyte" or "mono" or "mc"):
    generate_monocyte_images(args.count)
elif(args.type=="neutrophil" or "n", "np"):
    generate_neutrophil_images(args.count)
"""

def generate_all_images(count=1000, count_test=300):
    counter = 0
    test_counter = 0
    for i in range(0,300):
        image, yolobox = generate_neutrophil_image()
        plt.imsave("output/obj/img"+str(i)+".jpg",image, vmin=0, vmax=1)
        file = open("output/obj/img"+str(i)+".txt",'w')
        file.write("1 "+str(yolobox[0])+" "+str(yolobox[1])+ " " + str(yolobox[2])+ " " + str(yolobox[3]))
        file.close()
        counter+=1
        print("Progress: ",counter," out of ", count, " images, ",(counter/count)*100,"%")

    for i in range(count_test,count_test*2):
        image, yolobox = generate_neutrophil_image()
        plt.imsave("output/test/img"+str(i)+".jpg",image)
        file = open("output/test/img"+str(i)+".txt",'w')
        file.write("0 "+str(yolobox[0])+" "+str(yolobox[1])+ " " + str(yolobox[2])+ " " + str(yolobox[3]))
        file.close()
        test_counter+=1
        print("Progress: ",test_counter," out of ", count_test*2, " images, ",(test_counter/count_test*2)*100,"%") 

class BlobGenerationStrategy(ABC):
    @abstractmethod
    def generate_blob(self, size = 100):
        print("You did not input a strategy.")
        pass

class FourierBlobGenerationStrategy(BlobGenerationStrategy):
    def generate_blob(self, size=100):
        pts = [(random() + 0.8) * cmath.exp(2j*cmath.pi*i/7) for i in range(7)]
        pts = convexHull([(pt.real, pt.imag) for pt in pts])
        xs, ys = [interpolateSmoothly(zs, 30) for zs in zip(*pts)]

        # prazna slika size x size
        img = np.zeros((size, size))

        # prebacivanje u koordinate
        points = np.array(list(zip(xs, ys)))

        # normaliziranje da bude izmedju 0 i 1
        points += 2
        points /= 2 + 2

        # skaliranje na velicinu slike
        points *= size

        # openCV format
        points = points.astype(np.int32)
        points = points.reshape((-1, 1, 2))

        cv.fillPoly(img, [points], color=255)
        #plt.imshow(img, cmap="Greys")
        #plt.show()
        return img

class WBCGenerator():

    def __init__(self, blobGenerationStrategy: BlobGenerationStrategy) -> None:
        self._blobGenerationStrategy = blobGenerationStrategy

    @property
    def blobGenerationStrategy(self) -> BlobGenerationStrategy:
        return self._blobGenerationStrategy

    @blobGenerationStrategy.setter
    def blobGenerationStrategy(self, blobGenerationStrategy: BlobGenerationStrategy) -> None:
        self._blobGenerationStrategy = blobGenerationStrategy

    def generate_image(self, size = 100) -> None:
        img = self._blobGenerationStrategy.generate_blob(size)
        return img

if __name__ == "__main__":
    wbc_generator = WBCGenerator(FourierBlobGenerationStrategy)
    img = wbc_generator.generate_image(100)
    plt.imshow(img)
    plt.show()
    
