"""Glasses-aware compositing, with inspectable masks rather than identity claims."""
import cv2
import numpy as np


def glasses_layers(labels, image, face_width, policy='frames'):
    """Estimate a frame boundary band. This is not ground-truth frame segmentation.

    The parser's glasses label includes the transparent lenses. Dark, probably
    opaque glasses keep full protection because there is no visible eye to replace.
    """
    if policy not in ('full','frames'):
        raise ValueError('Glasses policy must be full or frames')
    mask=(labels==6).astype(np.uint8)
    empty=np.zeros_like(mask)
    if not mask.any():
        return empty,empty,False
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    opaque=bool(np.percentile(gray[mask>0],75)<55)
    radius=max(1,int(round(float(face_width)*.008)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    inside=cv2.erode(mask,kernel)
    outside=cv2.dilate(mask,kernel)
    frame=(outside & (1-inside)).astype(np.uint8)
    interior=cv2.erode(inside,kernel)
    protected=outside if policy=='full' or opaque else frame
    return protected,interior,opaque


def skin_only_color(generated, matched, labels_generated, labels_target):
    """Apply LAB adjustment on skin; do not recolor iris, teeth or lips."""
    skin=((labels_generated==1)&(labels_target==1)).astype(np.uint8)
    skin=cv2.erode(skin,np.ones((3,3),np.uint8))
    distance=cv2.distanceTransform(skin,cv2.DIST_L2,5)
    weight=np.clip(distance/3.,0,1)[...,None]
    result=np.rint(generated.astype(np.float32)*(1-weight)+matched.astype(np.float32)*weight)
    return np.clip(result,0,255).astype(np.uint8)


def bridge_protection(labels,image):
    """Conservative image-edge estimate in the gap between two lens components.

    Only the narrow upper gap is searched; iris regions inside the lenses are
    not copied wholesale. Thin/transparent bridges can still be missed.
    """
    glasses=(labels==6).astype(np.uint8)
    count,components,stats,centers=cv2.connectedComponentsWithStats(glasses)
    result=np.zeros_like(glasses)
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>100]
    if len(candidates)!=2:return result
    left,right=sorted(candidates,key=lambda i:centers[i,0])
    lx,ly,lw,lh=stats[left,:4];rx,ry,rw,rh=stats[right,:4]
    gap=rx-(lx+lw)
    if not 0<gap<max(lw,rw)*.8:return result
    width=min(lw,rw)
    x1=max(0,int(lx+lw-width*.13));x2=min(image.shape[1],int(rx+width*.13))
    center=int((ly+lh*.28+ry+rh*.28)/2)
    half=max(3,int(min(lh,rh)*.10))
    y1=max(0,center-half);y2=min(image.shape[0],center+half+1)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,25,65)
    result[y1:y2,x1:x2]=edges[y1:y2,x1:x2]>0
    return cv2.dilate(result,np.ones((3,3),np.uint8))
