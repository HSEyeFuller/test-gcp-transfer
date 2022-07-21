import numpy as np
from matplotlib import pyplot as plt
import heapq
import csv
from matplotlib import cm
import os
import shutil
from PIL import Image
import tensorflow as tf
import cv2

def showImage(image, dim = 144):
    data = np.reshape(image, (dim,dim,3))
    plt.imshow(imRegulate(data), interpolation='nearest')
    return plt.show()

def saveImage(image, fn, dim = 144):
    data = np.reshape(image, (dim,dim,3))
    plt.imshow(imRegulate(data), interpolation='nearest')
    plt.savefig(f'tmp/{fn}.png')

def showMap(depthMap, dim=0):

    d1 = depthMap.shape[0] if dim == 0 else dim
    d2 = depthMap.shape[1] if dim == 0 else dim
    
    pData = np.reshape(depthMap, (d1,d2))

    fig = plt.figure(figsize = (5,5))
    ax1 = fig.add_subplot(projection='3d')

    x = np.arange(0, d1)
    y = np.arange(0, d2)
    
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)

    ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
    ax1.set_title('Patient Tear Film Thickness Profile')



    return plt.show()

def saveMap(image, fn, dim = 144):
    pData = np.reshape(image, (dim,dim))
    
    fig = plt.figure(figsize = (5,5))
    ax1 = fig.add_subplot(projection='3d')
    ax1.set_zlim(0, 3000)
    x = np.arange(0, dim)
    y = np.arange(0, dim)
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)
    
    ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
    ax1.set_title('Patient Tear Film Thickness Profile')


    plt.savefig(f'tmp/{fn}.png')
    
def imRegulate(data):
    m = np.max(data)
    mi = np.min(data)
    norm = ((data - mi) / (m - mi))*255
    return norm.astype(np.uint8)

def collectVideoStats(imageSet):

    maxes = []
    mins = []
    for frame in imageSet:
        maxes.append(np.amax(frame))
        mins.append(np.amin(frame))
        
    return {'maxes': maxes, 'mins': mins}

def showVideoStats(imageSet):
    stats = collectVideoStats(imageSet)
    
    plt.scatter(np.arange(len(imageSet)), stats['maxes'])
    return plt.show()

def saveImages(imgSet):
 

    try:
        shutil.rmtree('tmp/')
    except:
        print("tmp/ does not exist")
    
    
    try:
        os. mkdir('tmp/')
    except:
        print("tmp/ exists")
        
    try:
        os.remove('tmp.zip')
    except:
        print("tmp.zip does not exist")
    for frame in range(imgSet.shape[0]):
        saveImage(imgSet[frame], frame)
        
    shutil.make_archive('tmp', 'zip', 'tmp/')

def saveMaps(imgSet):


    try:
        shutil.rmtree('tmp/')
    except:
        print("tmp/ does not exist")
    
    
    try:
        os. mkdir('tmp/')
    except:
        print("tmp/ exists")
        
    try:
        os.remove('tmp.zip')
    except:
        print("tmp.zip does not exist")
    for frame in range(imgSet.shape[0]):
        saveMap(imgSet[frame], frame)
        
    shutil.make_archive('tmp', 'zip', 'tmp/')

def saveEyeVideo(imgSet):
    saveImages(imgSet)
    image_folder = 'tmp'
    video_name = 'video1.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 4, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
def saveMapVideo(imgSet):
    saveMaps(imgSet)
    image_folder = 'tmp'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def auxSlice(content, channels):
    numVid = content.shape[0]
    numFrame = content.shape[1]
    width = content.shape[2]
    height = content.shape[3]
    
    
    slices = np.zeros((numVid, numFrame*width, height, channels))
        
    for i in range(numVid):
        stack = []
        for k in range(numFrame):
            stack.append(content[i][k].reshape(width, height, channels))
        slices[i] = np.vstack(stack)
        
        
    return slices

def imRegulate(data):
    m = np.max(data)
    mi = np.min(data)
    norm = ((data - mi) / (m - mi))*255
    return norm.astype(np.uint8)


def tRegulate(data):
    data = tf.truediv(
       tf.subtract(
          data, 
          tf.reduce_min(data)
       ), 
       tf.subtract(
          tf.reduce_max(data), 
          tf.reduce_min(data)
       )
    )
    return tf.math.multiply(data,5000)

@tf.custom_gradient
def tCycle(depth):
    depth = tRegulate(depth)
    depth = tf.convert_to_tensor(depth)
    depth = tf.cast(tf.reshape(depth, (48*48)), tf.int32).numpy()
    print(depth)

    def grad(upstream):
            values = dColorMap[[depth]]
            ret = tf.convert_to_tensor(values.reshape(1,48,48,3).astype(np.float32))
            return ret * upstream
    
    values = colorMap[[depth]]
    return tf.convert_to_tensor(values.reshape(1,48,48,3).astype(np.float32)), grad


def unpack(tiles, dim = 48):
    depth = round(tiles.shape[0]/dim)
    channels = tiles.shape[2]
    output = np.zeros((depth, dim, dim, channels))
    for i in range(depth):
        output[i] = tiles[i*dim:(i+1)*dim,:]


    return output