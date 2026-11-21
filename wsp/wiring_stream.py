"""Bounded, cancellable logical matrix reductions for the Systems graph.

All reads go through Ref so native permutations and current patch composition
remain identical to dense semantic views. No base-cache population or full-layer stack.
"""
import math
import time
import torch

SLAB_VALUES = 262144  # 1 MiB float32; doubles/projections are bounded temporaries
WORK_VALUES = 512_000_000
MEMORY_BYTES = 128*1024**2


class Window:
    def __init__(self, cancel=None, progress=None, seconds=120):
        self.cancel, self.progress = cancel, progress
        self.deadline = time.monotonic()+seconds
        self.values = self.max_read = 0
        self.last_progress = 0.

    def check(self):
        if self.cancel is not None and self.cancel.is_set(): raise ValueError('Wiring analysis cancelled')
        if time.monotonic()>self.deadline: raise ValueError('Wiring analysis reached its 120-second bound; choose fewer layers/components')

    def read(self, ref):
        self.check()
        if ref.size > SLAB_VALUES: raise ValueError('Wiring slab exceeds 1 MiB; unsupported residual/head width')
        self.values += ref.size
        if self.values > WORK_VALUES: raise ValueError('Wiring window exceeds 512M logical values; choose fewer layers')
        self.max_read=max(self.max_read,ref.size)
        out=ref.read(cache=False)
        if self.progress and time.monotonic()-self.last_progress>.5:
            self.progress(dict(values=self.values,max_values=WORK_VALUES))
            self.last_progress=time.monotonic()
        return out

    def slabs(self, ref, axis):
        if len(ref.shape)!=2: raise ValueError('Wiring requires a canonical matrix view')
        width=ref.shape[1-axis]
        if width>SLAB_VALUES: raise ValueError('Wiring matrix width exceeds bounded row minimum')
        step=max(1,SLAB_VALUES//width)
        for start in range(0,ref.shape[axis],step):
            index=[slice(None),slice(None)];index[axis]=slice(start,start+step)
            yield start,self.read(ref.sub(tuple(index)))


def rank_for(d,nwriters,nreaders):
    old=min(d,1024,max(192,int((6e7/max(1,nwriters))**.5)))
    # Includes both list + stack copies, Gram products, projection and slab/double buffers.
    def bound(r):return 8*(nwriters+nreaders+12)*r*r+8*d*r+16*SLAB_VALUES*4
    r=old
    while r>1 and bound(r)>MEMORY_BYTES:r=max(1,r-8)
    if bound(r)>MEMORY_BYTES:raise ValueError('Wiring component window exceeds retained-memory budget')
    return r,bound(r)


def writer_gram(ref,om,window):
    r=om.shape[1] if om is not None else ref.shape[1]
    gram=torch.zeros(r,r);norm=0.
    for _,block in window.slabs(ref,0):
        projected=block@om if om is not None else block
        gram.add_(projected.T@projected)
        norm+=float(block.double().square().sum())
    return gram,norm


def reader_gram(ref,om,window):
    r=om.shape[1] if om is not None else ref.shape[0]
    gram=torch.zeros(r,r);norm=0.
    for _,block in window.slabs(ref,1):
        projected=om.T@block if om is not None else block
        gram.add_(projected@projected.T)
        norm+=float(block.double().square().sum())
    return gram,norm


def head_writer(v,o,om,window):
    dh=o.shape[0];d=o.shape[1];r=om.shape[1] if om is not None else d
    if dh>512:raise ValueError('Wiring head dimension exceeds bounded 512-wide Gram support')
    gv=torch.zeros(dh,dh);oo=torch.zeros(dh,dh);op=torch.zeros(dh,r)
    for _,block in window.slabs(v,0):gv.add_(block.T@block)
    for start,block in window.slabs(o,1):
        oo.add_(block@block.T)
        if om is None:op[:,start:start+block.shape[1]]=block
        else:op.add_(block@om[start:start+block.shape[1]])
    return op.T@gv@op,float((gv*oo).sum()),float(oo.trace().sqrt())
