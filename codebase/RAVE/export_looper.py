from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

# import numpy as np
# from einops import rearrange

# from time import time
# import itertools as it

import cached_conv as cc
from rave.model import RAVE
from rave.core import search_for_run

from rave.weight_norm import remove_weight_norm

from effortless_config import Config

class args(Config):
    # run smoke tests before exporting
    TEST = 1
    # model checkpoint path
    CKPT = None
    # exported rave / rave+prior / neutone model
    # note that older models might not work (if they don't support batching)
    TS = None
    # audio sample rate -- currently must match RAVE model if given
    SR = 0
    # should be true for realtime
    CACHED = True
    # number of latents to preserve from RAVE model
    LATENT_SIZE = None
    # maximum predictive context size
    CONTEXT = 24
    # maximum number of frames to fit model on
    FIT = 150
    # number of loops
    LOOPS = 5
    # max frames for loop memory
    MEMORY = 1000
    # latency correction, latent frames
    LATENCY_CORRECT = 2
    # included in output filename
    NAME = "test"

args.parse_args()
cc.use_cached_conv(args.CACHED)


import logging
from termcolor import colored
logging.basicConfig(
    level=logging.INFO,
    format=colored("[%(relativeCreated).2f] ", "green") +
    "%(message)s")

class Loop(nn.Module):
    length:int
    end_step:int
    n_loops:int
    context:int
    feature_size:int

    def __init__(self, 
            index:int,
            n_loops:int,
            n_context:int, # maximum time dimension of model feature
            n_memory:int, # maximum loop memory 
            n_fit:int, # maximum dataset size to fit
            n_latent:int
            ):

        self.index = index
        self.n_loops = n_loops
        self.max_n_context = n_context # now a maximum
        self.n_memory = n_memory
        self.n_fit = n_fit
        self.n_latent = n_latent

        max_n_feature = n_loops * n_context * n_latent

        super().__init__()

        self.register_buffer('weights', 
            torch.empty(max_n_feature, n_latent, requires_grad=False))
        self.register_buffer('center', 
            torch.empty(max_n_feature, requires_grad=False))
        self.register_buffer('bias', 
            torch.empty(n_latent, requires_grad=False))

        # long-term memory stored at record-end
        self.register_buffer('memory', 
            torch.empty(n_memory, n_loops, n_latent, requires_grad=False))

    def reset(self):
        self.end_step = 0
        self.length = 0
        self.context = 0
        self.feature_size = 0

        self.memory.zero_()
        self.weights.zero_()
        self.bias.zero_()
        self.center.zero_()

    def feat_process(self, x, fit:bool=False):
        fs = x.shape[1]#self.feature_size
        if fit:
            c = self.center[:fs] = x.mean(0)
        else:
            c = self.center[:fs]
        return ((x - c)/2).tanh()

    def target_process(self, z):
        return torch.where(
            z > 1, ((z+1)/2)**2, torch.where(
                z < -1, -((1-z)/2)**2, z))

    def target_process_inv(self, z):
        return torch.where(
            z > 1, 2*z**0.5 - 1, torch.where(
                z < -1, 1 - 2*(-z)**0.5, z))

    def store(self, memory, step):
        """
        Args:
            memory: Tensor[time, loop, latent]
            step: int
        """
        self.length = memory.shape[0]
        self.end_step = step
        self.memory[:self.length] = memory

    def fit(self, feature, z):
        """
        Args:
            feature: Tensor[batch, context, loop, latent]
            z: Tensor[batch, latent]
        """
        # print(torch.linalg.vector_norm(feature, dim=(1,2,3)))
        # print(torch.linalg.vector_norm(feature, dim=(0,2,3)))
        # print(torch.linalg.vector_norm(feature, dim=(0,1,3)))
        # print(torch.linalg.vector_norm(feature, dim=(0,1,2)))

        self.context = feature.shape[1]
        assert feature.shape[2]==self.n_loops
        assert feature.shape[3]==self.n_latent
        print("[batch, context, loop, latent]: ", feature.shape)

        feature = feature.reshape(feature.shape[0], -1)
        self.feature_size = fs = feature.shape[1]
        
        z = self.target_process(z)

        feature = self.feat_process(feature, fit=True)

        # feature = feature + torch.randn_like(feature)*1e-7

        b = z.mean(0)
        self.bias[:] = b

        r = torch.linalg.lstsq(feature, z-b, driver='gelsd')
        w = r.solution
        self.weights[:fs] = w

        # print(torch.linalg.vector_norm(feature, dim=1))
        # print(w.norm())
        # print(torch.linalg.matrix_rank(feature))
        print("rank:", r.rank)
        # print(r.solution.shape, r.residuals, r.rank, r.singular_values)
        # print(feature.shape, (z-b).shape)
        # print(w.norm(), feature.norm(), (z-b).norm(), z-b)


    def eval(self, feature):
        """
        Args:
            feature: Tensor[context, loop, latent]
        Returns:
            Tensor[latent]
        """
        feature = feature[-self.context:].reshape(1,-1) 
        # 1 x (loop,latent,ctx)
        fs = feature.shape[1]#self.feature_size
        assert self.feature_size==0 or fs==self.feature_size

        w, b = self.weights[:fs], self.bias

        z = self.feat_process(feature) @ w + b

        z = self.target_process_inv(z).squeeze(0)

        return z

    def read(self, step:int):
        if self.length > 0:
            j = (step - self.end_step + self.latency_correct) % self.length
            z = self.memory[j, self.index]
        else:
            j = 0
            z = torch.zeros(self.n_latent)

        return z
        # if j < self.n_context:
            # return loop_z

        # # mix = float(j)/(loop_len-1)
        # mix = 1
        # # mix = min(1, (self.step - self.loop_end_step[i])/self.n_context)

        # return z*mix + loop_z*(1-mix)


class LivingLooper(nn.Module):
    __constants__ = ['loops']

    trained_cropped:bool
    sampling_rate:int
    block_size:int
    n_memory:int
    latency_correct:int

    loop_index:int
    record_index:int
    step:int
    record_length:int

    latency_correct:int

    def __init__(self, 
            rave_model:Union[RAVE, torch.jit.ScriptModule], 
            n_loops:int, 
            n_context:int, # maximum time dimension of model feature
            n_memory:int, # maximum loop memory 
            n_fit:int, # maximum dataset size to fit
            latency_correct:int, # in latent frames
            sr:int # sample rate
            ):
        super().__init__()

        self.n_loops = n_loops
        self.max_n_context = n_context # now a maximum
        self.n_memory = n_memory
        self.n_fit = n_fit

        self.min_loop = 2
        self.latency_correct = latency_correct

        # support standard exported RAVE models
        if isinstance(rave_model, torch.jit.ScriptModule):
            # unwrap neutone
            if hasattr(rave_model, 'model'):
                rave_model = rave_model.model
            # unwrap rave+prior
            if hasattr(rave_model, '_rave'):
                rave_model = rave_model._rave
            self.block_size = rave_model.encode_params[3].item()
            self.sampling_rate = rave_model.sampling_rate.item()
            self.rave = rave_model
        else:
            self.block_size = rave_model.block_size()
            self.sampling_rate = rave_model.hparams['sr']
            self.rave = None

        try:
            cropped_latent_size = rave_model.cropped_latent_size
        except AttributeError:
            cropped_latent_size = 0

        print(f'{self.block_size=}, {self.sampling_rate=}')

        assert sr==0 or sr==self.sampling_rate

        self.trained_cropped = bool(cropped_latent_size)
        self.n_latent = (
            rave_model.cropped_latent_size 
            if self.trained_cropped 
            else rave_model.latent_size)

        self.loops = nn.ModuleList(Loop(
            i, n_loops, n_context, n_memory, n_fit, self.n_latent
        ) for i in range(n_loops))

        self.pqmf = rave_model.pqmf
        self.encoder = rave_model.encoder
        self.decoder = rave_model.decoder
        self.n_latent_decoder = rave_model.latent_size

        # continuously updated last N frames of memory
        self.register_buffer('memory', 
            torch.empty(n_memory, n_loops, self.n_latent, requires_grad=False))

        self.register_buffer('mask', 
            torch.empty(2, n_loops, requires_grad=False))

        self.reset()

    @torch.jit.export
    def reset(self):
        self.record_length = 0
        self.step = 0
        self.loop_index = -1
        self.record_index = 0

        for l in self.loops:
            l.reset()

        self.memory.zero_()
        self.mask.zero_()

    def forward(self, i:int, x, oneshot:int=0):
        """
        Args:
            i: loop record index, 0 for no loop, 1-index otherwise
            x: Tensor[1, sample]
            oneshot: 0 for continuation, 1 to loop the training data
        Returns:
            Tensor[loop, sample]
        """
        self.step += 1
        lc = self.latency_correct
        # return self.decode(self.encode(x)) ### DEBUG

        # always encode for cache, even if result is not used
        z = self.encode(x) 

        if i > self.n_loops:
            i = 0
        # convert to zero index loop / negative for no loop
        i = i-1

        i_prev = self.loop_index
        # print(i, i_prev, self.loop_length)
        if i!=i_prev: # change in loop select control
            if i_prev >= 0: # previously on a loop
                if self.record_length >= self.min_loop: # and it was long enough
                    self.fit_loop(i_prev, oneshot)
            if i>=0: # starting a new loop recording
                self.record_length = 0
                self.mask[1,i] = 0.
            self.loop_index = i
                

        # eval the other loops
        mem = self.get_frames(self.max_n_context) # ctx x loop x latent

        # advance record head
        self.advance()

        # store encoded input to current loop
        if i>=0:
            # print(x.shape)
            # slice on LHS to appease torchscript
            z_i = z[0,:,0] # remove batch, time dims
            self.record(z_i, i, lc)

        # print(f'{feature.shape=}')
        for j,loop in enumerate(self.loops):
            # skip the loop which is now recording
            if i!=j:
                # construct feature for current loop
                feature = mem
                # feature = torch.cat((
                #     mem[:,j:j+1],
                #     mem[:,:j],
                #     mem[:,j+1:]
                # ), 1)
                z_j = loop.eval(feature)
                self.record(z_j, j, 0)

        # update memory
        # if self.loop_index >= 0:
        self.record_length += 1

        zs = self.get_frames(1).permute(1,2,0)
        y = self.decode(zs) # loops, channels (1), time

        fade = torch.linspace(0,1,y.shape[2])
        mask = self.mask[1,:,None,None] * fade + self.mask[0,:,None,None] * (1-fade)
        y = y * mask
        self.mask[0] = self.mask[1]

        return y

        # print(f'{self.loop_length}, {self.record_index}, {self.loop_index}')

        # return torch.stack((
        #     (torch.rand(self.block_size)-0.5)/3, 
        #     torch.zeros(self.block_size),
        #     (torch.arange(0,self.block_size)/128*2*np.pi).sin()/3 
        # ))[:,None]

    # def get_loop(self, i:int) -> Optional[Loop]:
    #     for j,loop in enumerate(self.loops):
    #         if i==j: return loop
    #     return None

    def fit_loop(self, i:int, oneshot:int):
        """
        assemble dataset and fit a loop to it
        """
        print(f'fit {i+1}')

        ll = min(self.n_memory, self.record_length)
        ctx = min(self.max_n_context, ll//2)
        lc = self.latency_correct

        # drop the last lc frames -- there are no target values
        # wrap the final n_context around
        # TODO: wrap target loop but not others?
        if oneshot:
            mem = self.get_frames(ll, lc)
            train_mem = torch.cat((mem[-ctx:], mem),0)
        else:
            mem = self.get_frames(ll+ctx, lc)
            train_mem = mem

        # limit to last n_fit frames
        # train_mem = train_mem[:self.n_fit+ctx]
        train_mem = train_mem[-(self.n_fit+ctx):]

        # dataset of features, targets
        features = train_mem.unfold(0, ctx, 1)[:-1] # batch, loop, latent, context
        features = features.permute(0,3,1,2).contiguous() # batch, context, loop, latent
        targets = train_mem[ctx:,i,:] # batch x latent

        # work around weird lacuna of torchscript
        # (can't index ModuleList except with literal)
        for j,loop in enumerate(self.loops): 
            if i==j:
                # loop.store(mem, self.step) # NOTE: disabled
                loop.fit(features, targets)
                # rollout predictions to make up latency
                for dt in range(lc,0,-1):
                    mem = self.get_frames(loop.context, dt)
                    z = loop.eval(mem)
                    self.record(z, j, dt-1)
                    
        self.mask[1,i] = 1.


    def advance(self):
        """
        advance the record head
        """
        self.record_index = (self.record_index+1)%self.n_memory

    def record(self, z, i:int, dt:int):
        """
        z: Tensor[latent]
        """
        t = (self.record_index-dt)%self.n_memory
        self.memory[t, i, :] = z

    def get_frames(self, n:int, skip:int=0):
        """
        get contiguous tensor out of ring memory
        """
        begin = self.record_index - n - skip + 1
        if begin < 0:
            begin1 = begin % self.n_memory
            block1 = self.memory[begin1:]
            begin2 = max(0, begin)
            block2 = self.memory[begin2:self.record_index+1]
            r = torch.cat((block1, block2))
        else:
            r = self.memory[begin:self.record_index+1]
        if skip>0:
            return r[:-skip]
        return r

    def encode(self, x):
        """
        RAVE encoder
        """
        if self.rave is not None:
            return self.rave.encode(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        z = self.encoder(x)[:,:self.n_latent]
        return z

    def decode(self, z):
        """
        RAVE decoder
        """
        if self.rave is not None:
            return self.rave.decode(z)

        pad_size = self.n_latent_decoder - self.n_latent
        pad_latent = torch.randn(
            z.shape[0],
            pad_size,
            z.shape[-1],
            device=z.device,
        )

        z = torch.cat([z, pad_latent], 1)

        x = self.decoder(z)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)

        return x
              

if args.CKPT is not None:
    logging.info("loading RAVE model from checkpoint")

    ckpt = search_for_run(args.CKPT)
    logging.info(f"using {ckpt}")

    # debug_kw = {}
    debug_kw = {'script':False}#, 'cropped_latent_size':36, 'latent_size':128} ###DEBUG
    # debug_kw = {'cropped_latent_size':8, 'latent_size':128} ###DEBUG

    model = RAVE.load_from_checkpoint(ckpt, **debug_kw, strict=False).eval()

    logging.info("flattening weights")
    for m in model.modules():
        if hasattr(m, "weight_g"):
            remove_weight_norm(m)

    if args.LATENT_SIZE is not None:
        model.crop_latent_space(int(args.LATENT_SIZE))

elif args.TS is not None:
    logging.info("loading RAVE model from torchscript")

    model = torch.jit.load(args.TS)

    if args.LATENT_SIZE is not None:
        logging.warn("torchscript models assumed to already be cropped")
        assert args.LATENT_SIZE in (model.cropped_latent_size, model.latent_size)

model.discriminator = None

logging.info("creating looper")
# ls = None if args.LATENT_SIZE is None else int(args.LATENT_SIZE)
looper = LivingLooper(model, 
    args.LOOPS, args.CONTEXT, 
    args.MEMORY, args.FIT,
    args.LATENCY_CORRECT,
    args.SR)
looper.eval()

# smoke test
def feed(i, oneshot=None):
    with torch.inference_mode():
        x = torch.rand(1, 1, 2**11)-0.5
        if oneshot is not None:
            looper(i, x, oneshot)
        else:
            looper(i, x)

def smoke_test():
    looper.reset()
    feed(0)
    for _ in range(31):
        feed(2)
    for _ in range(args.CONTEXT+3):
        feed(1, oneshot=1)
    for _ in range(10):
        feed(0)

    if args.TEST <= 1: return
    for _ in range(args.MEMORY+3):
        feed(3)
    feed(0)

logging.info("smoke test with pytorch")
if args.TEST > 0:
    smoke_test()

looper.reset()

logging.info("compiling torchscript")
looper = torch.jit.script(looper)

logging.info("smoke test with torchscript")
if args.TEST > 0:
    smoke_test()

looper.reset()

fname = f"ll_{args.NAME}.ts"
logging.info(f"saving '{fname}'")
looper.save(fname)
