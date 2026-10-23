import unittest
import cv2
import numpy as np
from faceswap.compositing import glasses_layers,skin_only_color,bridge_protection


class CompositingTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.ones((128,128),np.uint8)
        cv2.ellipse(self.labels,(64,64),(35,25),0,0,360,6,-1)
        self.image=np.full((128,128,3),150,np.uint8)

    def test_transparent_lens_center_is_not_protected(self):
        frame,lens,opaque=glasses_layers(self.labels,self.image,100,'frames')
        self.assertFalse(opaque)
        self.assertEqual(int(frame[64,64]),0)
        self.assertEqual(int(lens[64,64]),1)
        self.assertGreater(int(frame.sum()),0)

    def test_opaque_lenses_remain_protected(self):
        self.image[self.labels==6]=10
        frame,_,opaque=glasses_layers(self.labels,self.image,100,'frames')
        self.assertTrue(opaque)
        self.assertEqual(int(frame[64,64]),1)

    def test_legacy_protects_entire_lens(self):
        frame,_,_=glasses_layers(self.labels,self.image,100,'full')
        self.assertEqual(int(frame[64,64]),1)

    def test_absent_glasses_do_not_create_protection(self):
        frame,lens,opaque=glasses_layers(np.ones((128,128),np.uint8),self.image,100)
        self.assertFalse(frame.any())
        self.assertFalse(lens.any())
        self.assertFalse(opaque)

    def test_skin_color_does_not_recolor_features(self):
        labels=np.ones((128,128),np.uint8)
        labels[30:45,30:90]=4
        labels[85:95,40:80]=12
        before=np.full((128,128,3),100,np.uint8)
        after=np.full((128,128,3),180,np.uint8)
        output=skin_only_color(before,after,labels,labels)
        np.testing.assert_array_equal(output[labels!=1],before[labels!=1])
        self.assertEqual(int(output[60,60,0]),180)

    def test_disconnected_lenses_survive_component_filter(self):
        from types import SimpleNamespace
        from faceswap.pipeline import PortraitPipeline,Config
        labels=np.ones((512,512),np.uint8)
        for x in (175,335):cv2.ellipse(labels,(x,210),(50,35),0,0,360,6,-1)
        class Parser:
            def __call__(self,image):return labels
        pipeline=PortraitPipeline.__new__(PortraitPipeline)
        pipeline.config=Config(glasses_policy='frames')
        pipeline.parser=Parser()
        points=np.array([[80,80],[430,430]],np.float32)
        target={'crop':{'img_crop':np.full((512,512,3),150,np.uint8),
                        'M_o2c':np.eye(3,dtype=np.float32),'M_c2o':np.eye(3,dtype=np.float32)},
                'face':SimpleNamespace(landmark_2d_106=points)}
        result,alpha=pipeline.blend(np.full((512,512,3),200,np.uint8),
                                     np.full((512,512,3),150,np.uint8),target)
        self.assertGreater(float(alpha[210,175]),.99)
        self.assertGreater(float(alpha[210,335]),.99)
        protected=target['glasses_frame_crop']>0
        self.assertTrue(np.all(alpha[protected]==0))
        self.assertTrue(np.all(result[protected]==150))

    def test_bridge_edges_do_not_fill_lens_interiors(self):
        labels=np.ones((128,150),np.uint8)
        for x in (40,100):cv2.ellipse(labels,(x,60),(20,25),0,0,360,6,-1)
        image=np.full((128,150,3),180,np.uint8)
        cv2.line(image,(60,49),(80,49),(20,20,20),1)
        mask=bridge_protection(labels,image)
        self.assertTrue(mask[47:52,64:77].any())
        self.assertEqual(int(mask[60,40]),0)
        self.assertEqual(int(mask[60,100]),0)


if __name__=='__main__':unittest.main()
