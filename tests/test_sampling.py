import unittest
import cv2
import numpy as np
from faceswap.sampling import aligned_crop
from insightface.utils.face_align import estimate_norm

POINTS=np.array([[80,100],[150,100],[115,140],[90,180],[140,180]],np.float32)

class SamplingTests(unittest.TestCase):
    def test_native_reproduces_legacy_warp(self):
        image=np.random.default_rng(9).integers(0,256,(256,256,3),dtype=np.uint8)
        crop,matrix=aligned_crop(image,POINTS,128,'native')
        np.testing.assert_array_equal(crop,cv2.warpAffine(image,estimate_norm(POINTS,128),(128,128)))

    def test_area_keeps_transform_and_reduces_checkerboard_alias(self):
        pattern=(np.indices((1024,1024)).sum(axis=0)%2*255).astype(np.uint8)
        image=np.repeat(pattern[...,None],3,axis=2)
        points=POINTS*3.2
        old,m0=aligned_crop(image,points,128,'native')
        new,m1=aligned_crop(image,points,128,'area')
        np.testing.assert_allclose(m0,m1,atol=1e-10)
        self.assertLess(float(new[20:-20,20:-20].std()),float(old[20:-20,20:-20].std()))

    def test_area_avoids_extra_blur_when_upscaling(self):
        image=np.random.default_rng(1).integers(0,256,(64,64,3),dtype=np.uint8)
        a,_=aligned_crop(image,POINTS*.2,128,'area')
        b,_=aligned_crop(image,POINTS*.2,128,'native')
        np.testing.assert_array_equal(a,b)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):aligned_crop(np.zeros((256,256,3),np.uint8),POINTS,128,'x')

    def test_area_has_no_half_pixel_translation(self):
        y,x=np.indices((1024,1024),dtype=np.float32)
        ramp=np.stack([x,y,(x+y)/2],axis=-1)
        old,_=aligned_crop(ramp,POINTS*3.2,128,'native')
        new,_=aligned_crop(ramp,POINTS*3.2,128,'area')
        self.assertLess(float(np.abs(old[25:-25,25:-25]-new[25:-25,25:-25]).max()),.1)

if __name__=='__main__':unittest.main()
