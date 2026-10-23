"""Fetch an official FaceFusion 256-native candidate and retain release provenance."""
from pathlib import Path
import hashlib,json,time,zlib,requests
ROOT=Path(__file__).resolve().parents[2]
BASE='https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/'
name='hyperswap_1a_256'
dest=ROOT/'pretrained_weights/hyperswap'/f'{name}.onnx'
dest.parent.mkdir(parents=True,exist_ok=True)
expected=requests.get(BASE+name+'.hash',timeout=60)
expected.raise_for_status()
hash_text=expected.text.strip()
print('Publisher hash:',repr(hash_text),flush=True)
if not dest.exists():
    part=dest.with_suffix('.part')
    with requests.get(BASE+name+'.onnx',stream=True,timeout=(30,60)) as response:
        response.raise_for_status();total=response.headers.get('Content-Length');written=0;last=time.monotonic()
        print('Download bytes:',total,flush=True)
        with part.open('wb') as file:
            for block in response.iter_content(1024*1024):
                file.write(block);written+=len(block)
                if time.monotonic()-last>15:print(written,'bytes',flush=True);last=time.monotonic()
    part.replace(dest)
sha=hashlib.sha256();crc=0
with dest.open('rb') as file:
    for block in iter(lambda:file.read(1024*1024),b''):
        sha.update(block);crc=zlib.crc32(block,crc)
crc=f'{crc:08x}'
if len(hash_text)==8 and hash_text.lower()!=crc:raise RuntimeError('Publisher CRC32 mismatch')
if len(hash_text)==64 and hash_text.lower()!=sha.hexdigest():raise RuntimeError('Publisher SHA256 mismatch')
metadata={'url':BASE+name+'.onnx','publisher_hash':hash_text,'crc32':crc,'sha256':sha.hexdigest(),
          'bytes':dest.stat().st_size,'license':'ResearchRAIL-MS','native_size':256,
          'documentation':'https://github.com/facefusion/facefusion-labs/tree/master/hyperswap'}
dest.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(metadata,flush=True)
