"""Windows PDH counters: machine CPU utility and busiest physical GPU engine.

GPU process instances are SUMMED per engine before taking the busiest engine.
Unavailable counters are None, never a fabricated zero or CUDA allocator usage.
"""
import ctypes as C
from ctypes import wintypes as W
import re,time
from collections import defaultdict

class ValueUnion(C.Union):
    _fields_=[('doubleValue',C.c_double),('largeValue',C.c_longlong)]
class Value(C.Structure):
    _anonymous_=('value',)
    _fields_=[('status',W.DWORD),('value',ValueUnion)]
class Item(C.Structure):
    _fields_=[('name',W.LPWSTR),('value',Value)]

def aggregate_engines(values):
    grouped=defaultdict(float)
    for name,value in values:
        match=re.search(r'(luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+)_eng_(\d+)',name,re.I)
        if match:grouped[(match[1].lower(),match[2])]+=max(0.,value)
    adapters={}
    for (adapter,engine),value in grouped.items():
        adapters[adapter]=max(adapters.get(adapter,0.),min(100.,value))
    return adapters

class WindowsCounters:
    def __init__(self):
        self.pdh=C.WinDLL('pdh');p=self.pdh
        p.PdhOpenQueryW.argtypes=[W.LPCWSTR,C.c_size_t,C.POINTER(W.HANDLE)]
        p.PdhAddEnglishCounterW.argtypes=[W.HANDLE,W.LPCWSTR,C.c_size_t,C.POINTER(W.HANDLE)]
        p.PdhCollectQueryData.argtypes=[W.HANDLE]
        p.PdhGetFormattedCounterValue.argtypes=[W.HANDLE,W.DWORD,C.POINTER(W.DWORD),C.POINTER(Value)]
        p.PdhGetFormattedCounterArrayW.argtypes=[W.HANDLE,W.DWORD,C.POINTER(W.DWORD),C.POINTER(W.DWORD),C.c_void_p]
        p.PdhCloseQuery.argtypes=[W.HANDLE]
        self.query=W.HANDLE();self.handles={};self.errors={}
        if p.PdhOpenQueryW(None,0,C.byref(self.query)):raise OSError('PDH open failed')
        for name,path in {'cpu':r'\Processor Information(_Total)\% Processor Utility',
                          'gpu':r'\GPU Engine(*)\Utilization Percentage',
                          'memory':r'\GPU Adapter Memory(*)\Dedicated Usage'}.items():
            h=W.HANDLE();status=p.PdhAddEnglishCounterW(self.query,path,0,C.byref(h))
            if status==0:self.handles[name]=h
            else:self.errors[name]=hex(status & 0xffffffff)
        p.PdhCollectQueryData(self.query)

    def array(self,name):
        if name not in self.handles:return []
        size=W.DWORD();count=W.DWORD()
        h=self.handles[name];p=self.pdh
        p.PdhGetFormattedCounterArrayW(h,0x200,C.byref(size),C.byref(count),None)
        if not size.value:return []
        buffer=C.create_string_buffer(size.value)
        status=p.PdhGetFormattedCounterArrayW(h,0x200,C.byref(size),C.byref(count),buffer)
        if status:return []
        items=C.cast(buffer,C.POINTER(Item))
        return [(items[i].name,float(items[i].value.doubleValue)) for i in range(count.value)
                if items[i].value.status in (0,1)]

    def sample(self):
        status=self.pdh.PdhCollectQueryData(self.query);cpu=None
        if status==0 and 'cpu' in self.handles:
            value=Value()
            if self.pdh.PdhGetFormattedCounterValue(self.handles['cpu'],0x200,None,C.byref(value))==0 and value.status in (0,1):
                cpu=max(0.,min(100.,value.doubleValue))
        engines=self.array('gpu');adapters=aggregate_engines(engines)
        return {'timestamp':time.time(),'cpu':cpu,'gpu':max(adapters.values()) if adapters else None,
                'adapters':adapters,'dedicated_bytes':dict(self.array('memory')),
                'scope':'whole_machine_busiest_gpu_engine','source':'Windows PDH','errors':self.errors}

    def close(self):
        if self.query:self.pdh.PdhCloseQuery(self.query);self.query=None
