"""Aligned crops with an explicit, coordinate-preserving anti-aliasing option."""
import cv2
import numpy as np
from insightface.utils.face_align import estimate_norm


def aligned_crop(image, landmarks, size=128, mode='area'):
    if mode not in ('native','area','balanced','light'):
        raise ValueError('Input sampling must be native, light, balanced or area')
    matrix=estimate_norm(np.asarray(landmarks,np.float32),size)
    scale=float(np.linalg.norm(matrix[0,:2]))
    if mode=='native' or scale>=1:
        return cv2.warpAffine(image,matrix,(size,size)),matrix
    factor=4
    high_matrix=matrix.copy()*factor
    # Compensate resize's half-pixel centers; native and area share the same
    # original-to-native geometry rather than shifting the face by 0.375 pixel.
    high_matrix[:,2]+=(factor-1)/2
    high=cv2.warpAffine(image,high_matrix,(size*factor,size*factor),flags=cv2.INTER_LINEAR)
    area=cv2.resize(high,(size,size),interpolation=cv2.INTER_AREA)
    if mode in ('balanced','light'):
        native=cv2.warpAffine(image,matrix,(size,size))
        weight=.5 if mode=='balanced' else .25
        area=cv2.addWeighted(native,1-weight,area,weight,0)
    return area,matrix


def image_tensor(image, signed=False):
    tensor=cv2.dnn.blobFromImage(image,1/255.,(image.shape[1],image.shape[0]),(0,0,0),swapRB=True)
    return tensor*2-1 if signed else tensor
