"""Image-to-image portrait transfer; no intermediate JPEG or global argv changes."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
FACE_LABELS = (1, 2, 3, 4, 5, 10, 11, 12, 13)


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 32:
        raise ValueError(f"Unreadable or too small image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    data.tofile(path)


def feather(mask, radius):
    mask = (mask > 0).astype(np.uint8)
    if not mask.any():
        raise ValueError("Empty face mask")
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    x = np.clip(distance / max(radius, 1), 0, 1)
    return x * x * (3 - 2 * x)


def color_match(source, target, skin_a, skin_b, strength):
    if not 0 <= strength <= 1:
        raise ValueError("Color strength must be in [0, 1]")
    a = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    sa, sb = a[skin_a], b[skin_b]
    if min(len(sa), len(sb)) < 64:
        raise ValueError("Insufficient segmented skin for color matching")
    ma, mb = np.median(sa, axis=0), np.median(sb, axis=0)
    ra = np.percentile(sa, 75, axis=0) - np.percentile(sa, 25, axis=0)
    rb = np.percentile(sb, 75, axis=0) - np.percentile(sb, 25, axis=0)
    gain = np.clip((rb + 3) / (ra + 3), 0.8, 1.25)
    corrected = (a - ma) * gain + ma + np.clip(mb - ma, -35, 35)
    lab = np.clip(a + strength * (corrected - a), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@dataclass(frozen=True)
class Config:
    color_strength: float = 0.5
    feather_ratio: float = 0.035
    stitching: bool = False
    scale_policy: str = "source"
    device: str = "auto"
    seed: int = 42
    refine_alignment: bool = True
    retarget_lips: bool = True
    backend: str = 'inswapper'
    glasses_policy: str = 'frames'
    skin_color_only: bool = True
    direct_sampling: bool = True
    save_debug: bool = False
    input_sampling: str = 'native'
    protect_bridge: bool = False
    protection_mask: str | None = None

    def __post_init__(self):
        if not 0 <= self.color_strength <= 1 or not 0.005 <= self.feather_ratio <= 0.2:
            raise ValueError("Invalid color strength or feather ratio")
        if self.scale_policy not in ("source", "target"):
            raise ValueError("Invalid scale policy")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Invalid device")
        if self.backend not in ('liveportrait','inswapper','hyperswap'):
            raise ValueError('Invalid backend')
        if self.glasses_policy not in ('full','frames'):
            raise ValueError('Invalid glasses policy')
        if self.input_sampling not in ('native','area','balanced','light'):
            raise ValueError('Invalid input sampling')


def create_analyzer():
    from insightface.app import FaceAnalysis
    root = Path.home() / ".insightface"
    for name in ("det_10g.onnx", "2d106det.onnx", "1k3d68.onnx", "w600k_r50.onnx"):
        if not (root / "models/buffalo_l" / name).is_file():
            raise FileNotFoundError(f"Required local InsightFace model: {name}")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    # Explicit CPU fallback; Torch and ONNX need not have the same providers.
    app = FaceAnalysis(name="buffalo_l", root=str(root),
                       allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68", "recognition"],
                       providers=["CPUExecutionProvider"], sess_options=options)
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    return app


def select_face(app, image, index=0):
    faces = app.get(image)
    faces.sort(key=lambda f: float(np.prod(f.bbox[2:] - f.bbox[:2])), reverse=True)
    if index < 0 or index >= len(faces):
        raise ValueError(f"Face index {index} unavailable: detected {len(faces)} faces")
    face = faces[index]
    points = face.landmark_2d_106
    if points is None or points.shape != (106, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid face landmarks")
    return face, len(faces)


def corresponding_face(app, image, reference):
    faces = app.get(image)
    def overlap(face):
        a,b=face.bbox,reference.bbox
        intersection=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
        return intersection/max(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-intersection,1)
    if not faces:
        raise ValueError('No face detected in the final output')
    selected=max(faces,key=overlap)
    if overlap(selected)<.2:
        raise ValueError('Final output face does not overlap the selected target')
    return selected,len(faces)


class Parser:
    def __init__(self, device):
        from model import BiSeNet
        checkpoint = ROOT / "res/cp/79999_iter.pth"
        cache = Path(torch.hub.get_dir()) / "checkpoints/resnet18-5c106cde.pth"
        for path in (checkpoint, cache):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.device = device
        self.net = BiSeNet(19).to(device).eval()
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    @torch.inference_mode()
    def __call__(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].to(self.device) / 255
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        labels = self.net((x - mean) / std)[0].argmax(1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(labels, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


class PortraitPipeline:
    def __init__(self, config=None, analyzer=None):
        self.config = config or Config(backend='liveportrait')
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = "cuda:0" if self.config.device != "cpu" and torch.cuda.is_available() else "cpu"
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        self.wrapper = LivePortraitWrapper(InferenceConfig(flag_force_cpu=self.device == "cpu", flag_use_half_precision=False))
        torch.backends.cudnn.benchmark = False
        self.analyzer = analyzer or create_analyzer()
        self.parser = Parser(self.device)

    @torch.inference_mode()
    def prepare(self, image, index=0):
        from src.utils.crop import crop_image
        face, count = select_face(self.analyzer, image, index)
        crop = crop_image(image[..., ::-1].copy(), face.landmark_2d_106,
                          dsize=512, scale=2.3, vx_ratio=0, vy_ratio=-0.125, flag_do_rot=True)
        x = self.wrapper.prepare_source(cv2.resize(crop["img_crop"], (256, 256), interpolation=cv2.INTER_AREA))
        return {"face": face, "count": count, "crop": crop, "tensor": x, "info": self.wrapper.get_kp_info(x)}

    @torch.inference_mode()
    def render(self, source, target):
        from src.utils.camera import get_rotation_matrix
        s, t = source["info"], target["info"]
        rotation = get_rotation_matrix(t["pitch"], t["yaw"], t["roll"])
        translation = t["t"].clone()
        translation[..., 2] = 0
        scale = t["scale"] if self.config.scale_policy == "target" else s["scale"]
        destination = scale * (s["kp"] @ rotation + t["exp"]) + translation
        original = self.wrapper.transform_keypoint(s)
        if self.config.stitching:
            destination = self.wrapper.stitching(original, destination)
        feature = self.wrapper.extract_feature_3d(source["tensor"])
        out = self.wrapper.warp_decode(feature, original, destination)["out"]
        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite neural output")
        if self.config.retarget_lips:
            # Measure the synthesized mouth before correcting it; using A's
            # original aperture here would apply an incorrect second transfer.
            from src.utils.human_landmark_runner import LandmarkRunner
            from src.utils.retargeting_utils import calc_lip_close_ratio
            from src.utils.crop import _transform_pts
            if not hasattr(self, '_landmark_runner'):
                self._landmark_runner = LandmarkRunner(
                    ckpt_path=str(ROOT/'pretrained_weights/liveportrait/landmark.onnx'), onnx_provider='cpu')
            generated_rgb = self.wrapper.parse_output(out)[0]
            generated_face,_ = select_face(self.analyzer,generated_rgb[...,::-1].copy())
            generated_lmk = self._landmark_runner.run(generated_rgb,generated_face.landmark_2d_106)
            target_lmk = self._landmark_runner.run(target['crop']['img_crop'],
                _transform_pts(target['face'].landmark_2d_106,target['crop']['M_o2c']))
            measured = calc_lip_close_ratio(generated_lmk[None])
            desired = calc_lip_close_ratio(target_lmk[None])
            combined = torch.from_numpy(np.concatenate([measured,desired],axis=1)).float().to(self.device)
            destination = destination + self.wrapper.retarget_lip(destination,combined)
            out = self.wrapper.warp_decode(feature,original,destination)['out']
        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite neural output")
        return self.wrapper.parse_output(out)[0][..., ::-1].copy()

    def blend(self, generated, target_image, target):
        crop_bgr = target["crop"]["img_crop"][..., ::-1].copy()
        if self.config.refine_alignment and self.config.backend=='liveportrait':
            # 3D synthesis transfers yaw/pitch; a residual similarity fit fixes
            # crop scale/translation without fitting identity-specific face shape.
            generated_face, _ = select_face(self.analyzer, generated)
            from src.utils.crop import _transform_pts
            target_points = _transform_pts(target["face"].kps, target["crop"]["M_o2c"])
            matrix, inliers = cv2.estimateAffinePartial2D(
                generated_face.kps.astype(np.float32), target_points.astype(np.float32),
                method=cv2.LMEDS)
            if matrix is None or not np.isfinite(matrix).all():
                raise ValueError("Cannot refine generated face alignment")
            scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
            if not 0.5 <= scale <= 2:
                raise ValueError("Unreliable residual face alignment")
            generated = cv2.warpAffine(generated, matrix, (512,512), flags=cv2.INTER_CUBIC)
        lg, lt = self.parser(generated), self.parser(crop_bgr)
        target['glasses_mask_crop']=(lt==6).astype(np.uint8)
        from .compositing import glasses_layers, skin_only_color,bridge_protection
        face_width=float(np.ptp(target['face'].landmark_2d_106[:,0]))
        # Radius in crop coordinates, not original image pixels.
        face_width*=float(np.linalg.norm(target['crop']['M_o2c'][0,:2]))
        frame,lens_interior,opaque=glasses_layers(lt,crop_bgr,face_width,self.config.glasses_policy)
        if self.config.protect_bridge and self.config.glasses_policy=='frames' and not opaque:
            frame|=bridge_protection(lt,crop_bgr)
        if self.config.protection_mask:
            custom=cv2.imdecode(np.fromfile(self.config.protection_mask,np.uint8),cv2.IMREAD_GRAYSCALE)
            if custom is None or custom.shape!=target_image.shape[:2]:
                raise ValueError('Protection mask must match the original target dimensions')
            # This is an explicit opaque protection mask, not an optical lens layer.
            frame|=(cv2.warpAffine(custom,target['crop']['M_o2c'][:2],(512,512),flags=cv2.INTER_NEAREST)>127).astype(np.uint8)
            target['custom_protection_original']=custom>127
        target['glasses_frame_crop']=frame
        target['lens_interior_crop']=lens_interior
        target['opaque_glasses']=opaque
        allowed=FACE_LABELS+(6,) if self.config.glasses_policy=='frames' else FACE_LABELS
        support = np.isin(lg, allowed) & np.isin(lt, allowed)
        if 'generated_valid_mask' in target:
            support &= target['generated_valid_mask'] > .99
        blockers = np.isin(lt, (7, 8, 9, 14, 15, 16, 17, 18)).astype(np.uint8)
        blockers=cv2.dilate(blockers,np.ones((5,5),np.uint8))
        if self.config.glasses_policy=='full':
            blockers|=cv2.dilate((lt==6).astype(np.uint8),np.ones((5,5),np.uint8))
        else:
            blockers|=frame
        support &= blockers == 0
        # Fill tiny segmentation gaps; re-apply target occluders after closing.
        support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        support &= (blockers == 0).astype(np.uint8)
        n, components, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8))
        if n <= 1:
            raise ValueError("No common face region")
        # A protected frame disconnects the lens interiors from the rest of the
        # face. Keep those valid islands rather than reverting both eyes to B.
        lens_support=support & (lt==6).astype(np.uint8) if self.config.glasses_policy=='frames' else np.zeros_like(support)
        support = (components == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        support |= lens_support
        if support.sum() < 256:
            raise ValueError("Face mask too small")
        box = cv2.boundingRect(support)
        alpha_crop = feather(support, min(box[2:]) * self.config.feather_ratio)
        matched = color_match(generated, crop_bgr, (lg == 1) & (alpha_crop > 0.9),
                              (lt == 1) & (alpha_crop > 0.9), self.config.color_strength)
        if self.config.skin_color_only:
            matched=skin_only_color(generated,matched,lg,lt)
        matrix = target["crop"]["M_c2o"][:2]
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
            raise ValueError("Invalid crop transform")
        size = (target_image.shape[1], target_image.shape[0])
        premult = cv2.warpAffine(matched.astype(np.float32) * alpha_crop[..., None], matrix, size)
        alpha = np.clip(cv2.warpAffine(alpha_crop, matrix, size), 0, 1)
        if self.config.glasses_policy=='frames':
            protected_original=cv2.warpAffine(frame,matrix,size,flags=cv2.INTER_NEAREST)>0
            alpha[protected_original]=0
            premult[protected_original]=0
        if 'custom_protection_original' in target:
            alpha[target['custom_protection_original']]=0
            premult[target['custom_protection_original']]=0
        if self.config.direct_sampling and 'native_generated' in target:
            # Sample the native model image once; do not upsample to 512 then
            # resample the face again just to return to original coordinates.
            native=cv2.warpAffine(target['native_generated'],target['native_to_original'],size,
                                 flags=cv2.INTER_CUBIC).astype(np.float32)
            delta=matched.astype(np.float32)-generated.astype(np.float32)
            correction=cv2.warpAffine(delta,matrix,size)
            premult=np.clip(native+correction,0,255)*alpha[...,None]
        result = np.clip(np.rint(premult + target_image.astype(np.float32) * (1-alpha[..., None])), 0, 255).astype(np.uint8)
        result[alpha == 0] = target_image[alpha == 0]
        target['debug']={'target_crop':crop_bgr,'generated_crop':generated,'matched_crop':matched,
                         'labels_generated':lg,'labels_target':lt,'support':support,
                         'alpha_crop':alpha_crop,'frame_protection':frame,'lens_interior':lens_interior,
                         'result_crop':cv2.warpAffine(result,target['crop']['M_o2c'][:2],(512,512))}
        return result, alpha

    def swap(self, source_path, target_path, output_path, source_index=0, target_index=0, progress=None):
        def emit(stage, state):
            if progress is not None:progress(stage,state)
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".png":
            raise ValueError("Use PNG output to preserve untouched target pixels")
        outputs = [output_path, output_path.with_suffix('.json'),
                   output_path.with_name(output_path.stem + '_mask.png')]
        inputs = {Path(source_path).resolve(), Path(target_path).resolve()}
        if self.config.protection_mask:inputs.add(Path(self.config.protection_mask).resolve())
        if any(path.resolve() in inputs for path in outputs):
            raise ValueError("Output must not overwrite an input")
        if self.config.save_debug:
            debug_directory=output_path.with_name(output_path.stem+'_debug').resolve()
            if any(path.is_relative_to(debug_directory) for path in inputs):
                raise ValueError('Debug output directory must not contain an input')
        started = time.perf_counter()
        emit('read','started')
        a, b = read_image(source_path), read_image(target_path)
        emit('read','completed')
        emit('source','started')
        source = self.prepare(a, source_index)
        emit('source','completed')
        emit('target','started')
        target = self.prepare(b, target_index)
        emit('target','completed')
        emit('generate','started')
        generated = self.render(source, target)
        emit('generate','completed')
        emit('blend','started')
        result, alpha = self.blend(generated, b, target)
        emit('blend','completed')
        emit('validate','started')
        output_face, output_count = corresponding_face(self.analyzer,result,target['face'])
        from .quality import diagnostics
        hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (source_path,target_path)]
        quality = diagnostics(source['face'],target['face'],output_face,same_input=hashes[0]==hashes[1])
        quality['outside_alpha_max_error'] = int(np.abs(result[alpha==0].astype(int)-b[alpha==0].astype(int)).max()) if np.any(alpha==0) else None
        glasses=cv2.warpAffine(target['glasses_mask_crop'],target['crop']['M_c2o'][:2],
                               (b.shape[1],b.shape[0]),flags=cv2.INTER_NEAREST)>0
        quality['target_glasses_pixel_count']=int(glasses.sum())
        quality['target_glasses_max_error']=int(np.abs(result[glasses].astype(int)-b[glasses].astype(int)).max()) if glasses.any() else None
        frames=cv2.warpAffine(target['glasses_frame_crop'],target['crop']['M_c2o'][:2],
                              (b.shape[1],b.shape[0]),flags=cv2.INTER_NEAREST)>0
        lens=cv2.warpAffine(target['lens_interior_crop'],target['crop']['M_c2o'][:2],
                            (b.shape[1],b.shape[0]),flags=cv2.INTER_NEAREST)>0
        quality['estimated_frame_pixel_count']=int(frames.sum())
        quality['estimated_frame_max_error']=int(np.abs(result[frames].astype(int)-b[frames].astype(int)).max()) if frames.any() else None
        quality['lens_mean_generated_alpha']=float(alpha[lens].mean()) if lens.any() else None
        quality['opaque_glasses']=target['opaque_glasses']
        quality['visual_quality_status']='unverified_requires_human_review'
        if 'custom_protection_original' in target:
            custom=target['custom_protection_original']
            quality['custom_protection_pixel_count']=int(custom.sum())
            quality['custom_protection_max_error']=int(np.abs(result[custom].astype(int)-b[custom].astype(int)).max()) if custom.any() else None
        quality['postprocess_invariants_pass']=bool(quality['outside_alpha_max_error']==0 and
            (self.config.glasses_policy!='frames' or quality['estimated_frame_max_error'] in (None,0)))
        quality['frame_mask_method']='semantic_boundary_estimate_not_ground_truth'
        emit('validate','completed')
        emit('save','started')
        write_image(output_path, result)
        write_image(output_path.with_name(output_path.stem + "_mask.png"), np.rint(alpha*255).astype(np.uint8))
        meta = {"source": str(Path(source_path).resolve()), "target": str(Path(target_path).resolve()),
                "config": asdict(self.config), "device": self.device, "onnx_execution": "CPUExecutionProvider",
                "available_onnx_providers": ort.get_available_providers(), "face_counts": [source["count"], target["count"]],
                "selected_indices": [source_index, target_index], "seconds": time.perf_counter()-started,
                "input_sha256": hashes, 'output_face_count':output_count, 'quality':quality}
        meta['pipeline_revision']='sampling-v3'
        meta['implementation_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                        for name in ('pipeline.py','compositing.py','quality.py','sampling.py')}
        if 'native_generated' in target:
            meta['native_generator_shape']=list(target['native_generated'].shape)
        if hasattr(self,'model_sha256'):
            meta['identity_model_sha256']=self.model_sha256
        meta['visual_acceptance']={'decision':'pending_human_review',
            'checks':{key:'pending' for key in ('source_resemblance','eyes_mouth','skin_texture','seams',
                      'frame_rims','glasses_bridge','lens_reflections')},
            'automatic_scores_are_not_visual_approval':True}
        if self.config.protection_mask:
            meta['protection_mask_sha256']=hashlib.sha256(Path(self.config.protection_mask).read_bytes()).hexdigest()
        if self.config.save_debug:
            debug_dir=output_path.with_name(output_path.stem+'_debug')
            self.save_debug(target,debug_dir)
            meta['debug_directory']=str(debug_dir.resolve())
        output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        emit('save','completed')
        return result, alpha, meta

    @staticmethod
    def save_debug(target,directory):
        directory=Path(directory)
        for name,image in target['debug'].items():
            if name.startswith('labels_'):
                image=(image.astype(np.uint16)*12).astype(np.uint8)
            elif image.ndim==2:
                image=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
            write_image(directory/(name+'.png'),image)
        if 'native_generated' in target:
            write_image(directory/'model_native.png',target['native_generated'])
        if 'model_input_native' in target:
            write_image(directory/'model_input_native.png',target['model_input_native'])
        transforms={name:target['crop'][name].tolist() for name in ('M_o2c','M_c2o')}
        if 'native_to_original' in target:
            transforms['native_to_original']=target['native_to_original'].tolist()
        (directory/'transforms.json').write_text(json.dumps(transforms,indent=2),encoding='utf-8')


class IdentityPipeline(PortraitPipeline):
    """Identity-conditioned model: B supplies pose/expression, A supplies an embedding."""
    def __init__(self,config=None,analyzer=None,parser=None):
        from insightface.model_zoo.inswapper import INSwapper
        self.config=config or Config(backend='inswapper')
        if self.config.device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.device='cuda:0' if self.config.device!='cpu' and torch.cuda.is_available() else 'cpu'
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.benchmark=False
        self.analyzer=analyzer or create_analyzer()
        self.parser=parser or Parser(self.device)
        model_path=ROOT/'pretrained_weights/inswapper/inswapper_128.onnx'
        if not model_path.is_file():
            raise FileNotFoundError(f'{model_path}; run experiments/faceswap/download_swapper.py')
        with model_path.open('rb') as file:
            self.model_sha256=hashlib.file_digest(file,'sha256').hexdigest()
        if self.model_sha256!='e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af':
            raise ValueError('Identity model digest mismatch')
        options=ort.SessionOptions()
        options.intra_op_num_threads,options.inter_op_num_threads=4,1
        session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
        self.swapper=INSwapper(model_file=str(model_path),session=session)

    def prepare(self,image,index=0):
        from src.utils.crop import crop_image
        face,count=select_face(self.analyzer,image,index)
        crop=crop_image(image[...,::-1].copy(),face.landmark_2d_106,
                        dsize=512,scale=2.3,vx_ratio=0,vy_ratio=-.125,flag_do_rot=True)
        return {'face':face,'count':count,'crop':crop,'image':image}

    def render(self,source,target):
        from .sampling import aligned_crop,image_tensor
        model_input,matrix=aligned_crop(target['image'],target['face'].kps,128,self.config.input_sampling)
        latent=source['face'].normed_embedding.reshape(1,-1)@self.swapper.emap
        norm=np.linalg.norm(latent)
        if not np.isfinite(latent).all() or norm<1e-8:
            raise ValueError('Invalid source identity embedding')
        latent=latent/norm
        prediction=self.swapper.session.run(self.swapper.output_names,
            {self.swapper.input_names[0]:image_tensor(model_input),self.swapper.input_names[1]:latent})[0]
        if not np.isfinite(prediction).all():raise RuntimeError('Non-finite identity model output')
        generated=np.clip(prediction.transpose(0,2,3,1)[0]*255,0,255).astype(np.uint8)[...,::-1].copy()
        target['model_input_native']=model_input
        target['native_generated']=generated.copy()
        target['native_to_original']=cv2.invertAffineTransform(matrix)
        inverse=np.vstack([cv2.invertAffineTransform(matrix),[0,0,1]])
        to_crop=(target['crop']['M_o2c']@inverse)[:2].astype(np.float32)
        target['generated_valid_mask']=cv2.warpAffine(np.ones(generated.shape[:2],np.float32),to_crop,(512,512))
        return cv2.warpAffine(generated,to_crop,(512,512),flags=cv2.INTER_CUBIC)


class HyperSwapPipeline(IdentityPipeline):
    """Experimental 256-native comparison backend; compositor is shared."""
    def __init__(self,config=None,analyzer=None,parser=None):
        self.config=config or Config(backend='hyperswap')
        if self.config.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        self.device='cuda:0' if self.config.device!='cpu' and torch.cuda.is_available() else 'cpu'
        self.analyzer=analyzer or create_analyzer()
        self.parser=parser or Parser(self.device)
        path=ROOT/'pretrained_weights/hyperswap/hyperswap_1a_256.onnx'
        with path.open('rb') as file:self.model_sha256=hashlib.file_digest(file,'sha256').hexdigest()
        if self.model_sha256!='c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add':
            raise ValueError('HyperSwap model digest mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def render(self,source,target):
        from .sampling import aligned_crop,image_tensor
        image,matrix=aligned_crop(target['image'],target['face'].kps,256,self.config.input_sampling)
        identity=source['face'].normed_embedding.reshape(1,-1).astype(np.float32)
        output=self.session.run(['output'],{'source':identity,'target':image_tensor(image,signed=True)})[0]
        if output.shape!=(1,3,256,256) or not np.isfinite(output).all():raise RuntimeError('Invalid HyperSwap output')
        generated=np.clip((output[0].transpose(1,2,0)*.5+.5)*255,0,255).astype(np.uint8)[...,::-1].copy()
        target['model_input_native']=image
        target['native_generated']=generated
        target['native_to_original']=cv2.invertAffineTransform(matrix)
        inverse=np.vstack([target['native_to_original'],[0,0,1]])
        to_crop=(target['crop']['M_o2c']@inverse)[:2].astype(np.float32)
        target['generated_valid_mask']=cv2.warpAffine(np.ones((256,256),np.float32),to_crop,(512,512))
        return cv2.warpAffine(generated,to_crop,(512,512),flags=cv2.INTER_CUBIC)


def create_pipeline(config=None):
    config=config or Config()
    if config.backend=='hyperswap':return HyperSwapPipeline(config)
    return IdentityPipeline(config) if config.backend=='inswapper' else PortraitPipeline(config)
