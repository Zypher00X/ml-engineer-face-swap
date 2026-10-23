import tempfile
import unittest
from pathlib import Path
import numpy as np
from faceswap.pipeline import Config, read_image, write_image, feather, color_match, select_face, corresponding_face


class PortraitTests(unittest.TestCase):
    def test_unicode_png_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ภาพทดสอบ.png'
            image = np.random.default_rng(42).integers(0,256,(47,63,3),dtype=np.uint8)
            write_image(path,image)
            np.testing.assert_array_equal(read_image(path),image)

    def test_missing_and_corrupt_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.png'
            with self.assertRaises(FileNotFoundError): read_image(path)
            path.write_bytes(b'not an image')
            with self.assertRaises(ValueError): read_image(path)

    def test_feather_does_not_leak(self):
        for size in (32,128,512):
            mask=np.zeros((size,size),np.uint8)
            mask[size//4:3*size//4,size//4:3*size//4]=1
            alpha=feather(mask,size*.035)
            self.assertTrue(np.all(alpha[mask==0]==0))
            self.assertEqual(float(alpha.max()),1)
            self.assertTrue(np.isfinite(alpha).all())
        with self.assertRaises(ValueError): feather(np.zeros((32,32),np.uint8),3)

    def test_skin_stats_require_samples(self):
        image=np.full((64,64,3),120,np.uint8)
        empty=np.zeros((64,64),bool)
        with self.assertRaises(ValueError): color_match(image,image,empty,empty,.5)
        skin=np.ones((64,64),bool)
        output=color_match(image,image,skin,skin,.5)
        self.assertLessEqual(int(np.abs(output.astype(int)-image).max()),2)

    def test_configuration_bounds(self):
        for kwargs in ({'color_strength':2},{'feather_ratio':0},{'device':'bad'},{'scale_policy':'bad'}):
            with self.assertRaises(ValueError): Config(**kwargs)

    def test_no_face(self):
        class Empty:
            def get(self,image): return []
        with self.assertRaisesRegex(ValueError,'detected 0'): select_face(Empty(),np.zeros((64,64,3),np.uint8))

    def test_output_face_matching_uses_overlap(self):
        from types import SimpleNamespace
        left=SimpleNamespace(bbox=np.array([0,0,100,100],np.float32))
        right=SimpleNamespace(bbox=np.array([200,0,400,200],np.float32))
        class App:
            def get(self,image): return [right,left]
        selected,count=corresponding_face(App(),None,left)
        self.assertIs(selected,left)
        self.assertEqual(count,2)

    def test_identity_gate_for_identical_input(self):
        from types import SimpleNamespace
        from faceswap.quality import diagnostics
        face=SimpleNamespace(embedding=np.ones(512),pose=np.zeros(3),
                             landmark_3d_68=np.arange(204).reshape(68,3).astype(float))
        self.assertTrue(diagnostics(face,face,face,same_input=True)['project_gate_pass'])
        self.assertFalse(diagnostics(face,face,face,same_input=False)['project_gate_pass'])

    def test_debug_export_cannot_overwrite_an_input(self):
        from faceswap.pipeline import PortraitPipeline
        p=PortraitPipeline.__new__(PortraitPipeline)
        p.config=Config(save_debug=True)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'Debug output directory'):
                p.swap(root/'output_debug/model_native.png',root/'target.png',root/'output.png')

    def test_progress_does_not_complete_a_failed_model_stage(self):
        from faceswap.pipeline import PortraitPipeline
        p=PortraitPipeline.__new__(PortraitPipeline);p.config=Config()
        p.prepare=lambda image,index:{}
        def fail(source,target):raise RuntimeError('model failed')
        p.render=fail;events=[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'input.png'
            write_image(path,np.full((64,64,3),127,np.uint8))
            with self.assertRaisesRegex(RuntimeError,'model failed'):
                p.swap(path,path,root/'out.png',progress=lambda stage,state:events.append((stage,state)))
            self.assertEqual(events[-1],('generate','started'))
            self.assertNotIn(('generate','completed'),events)
            self.assertFalse((root/'out.png').exists())


if __name__=='__main__': unittest.main()
