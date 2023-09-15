import os
# fix torch device order to be same as nvidia-smi order
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

from pathlib import Path
import random
from collections import defaultdict
import itertools as it

from tqdm import tqdm
import fire

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import TacotronDecoder
from util import deep_update, get_class_defaults, JSONDataset, ConcatSpeakers

class Trainer:
    def __init__(self, 
        experiment, # experiment name
        model_dir, # to store checkpoints
        log_dir, # to store tensorboard logs
        manifest, # hifitts-style json
        csv=None, # optional CSV metadata
        concat_speakers=0,
        strip_quotes=False,
        rave_model = None,
        results_dir = None,
        model = None, # dict of model constructor overrides
        freeze_text = False, # freeze text encoder
        batch_size = 32,
        # TODO: specify estimated tokens / frame ? 
        # max number of audio and text elements to crop to:
        batch_max_tokens = 128,
        batch_max_frames = 512, # should ~always result in longer audio than text
        lr = 1e-4,
        adam_betas = (0.9, 0.998),
        adam_eps = 1e-08, 
        weight_decay = 1e-6,
        grad_clip = 5.0,
        seed = 0, # random seed
        n_jobs = 1, # for dataloaders
        device = 'cpu', # 'cuda:0'
        epoch_size = None, # in iterations, None for whole dataset
        valid_size = None, # in batches
        save_epochs = 1,
        nll_scale = 1, # scale nll loss
        # anneal_prenet = None, # number of epochs to anneal prenet dropout to zero
        ):
        """TODO: Trainer __init__ docstring"""
        kw = locals(); kw.pop('self')

        # store all hyperparams for checkpointing
        self.kw = kw

        self.best_step = None
        self.best_loss = np.inf

        # get model defaults from model class
        model_cls = TacotronDecoder
        if model is None: model = {}
        assert isinstance(model, dict), """
            model keywords are not a dict. check shell/fire syntax
            """
        kw['model'] = model = get_class_defaults(model_cls) | model

        # assign all arguments to self by default
        self.__dict__.update(kw)
        # mutate some arguments:
        self.model_dir = Path(model_dir) / self.experiment
        self.log_dir = Path(log_dir) / self.experiment
        if results_dir is None:
            self.results_dir = None
        else:
            self.results_dir = Path(results_dir) / self.experiment
        self.manifest = Path(manifest)
        self.device = torch.device(device)

        # filesystem
        for d in (self.model_dir, self.log_dir, self.results_dir):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)

        if rave_model is None:
            self.rave_model = None
        else:
            self.rave_model = torch.jit.load(rave_model)

        # random states
        self.seed_random()

        # logging
        self.writer = SummaryWriter(self.log_dir)

        # Trainer state
        self.iteration = 0
        self.exposure = 0
        self.epoch = 0

        # construct model from arguments 
        self.model = model_cls(**model).to(self.device)
        tqdm.write(repr(self.model))

        # dataset
        if concat_speakers > 0:
            self.train_dataset, self.valid_dataset, self.test_dataset = (
                 ConcatSpeakers(
                    [self.manifest]*concat_speakers, 
                    self.batch_max_tokens, self.batch_max_frames, split)
                for split in ('train', 'valid', 'test'))
            self.valid_size = valid_size
        else:
            self.dataset = JSONDataset(
                self.manifest, csv, self.batch_max_tokens, self.batch_max_frames,
                strip_quotes=strip_quotes, rave=self.rave_model)
            valid_len = max(8, int(len(self.dataset)*0.03))
            test_len = max(8, int(len(self.dataset)*0.02))
            train_len = len(self.dataset) - valid_len - test_len
            self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset, [train_len, valid_len, test_len], 
                generator=torch.Generator().manual_seed(0))
            self.valid_size = valid_size or valid_len//batch_size

        if freeze_text:
            self.model.text_encoder.requires_grad_(False)
            
        self.opt = torch.optim.AdamW(self.model.parameters(),
            self.lr, self.adam_betas, self.adam_eps, self.weight_decay)
        

    @property
    def gpu(self):
        return self.device.type!='cpu'

    def seed_random(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def set_random_state(self, states):
        # note: GPU rng state not handled
        std_state, np_state, torch_state = states
        random.setstate(std_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    @property
    def step(self):
        return self.exposure, self.iteration, self.epoch

    def save(self, fname):
        torch.save(dict(
            kw=self.kw,
            model_state=self.model.state_dict(),
            optimizer_state=self.opt.state_dict(),
            step=self.step,
            best_loss=self.best_loss,
            best_step=self.best_step,
            random_state=(random.getstate(), np.random.get_state(), torch.get_rng_state())
        ), fname)

    def load_state(self, d, resume):
        d = d if hasattr(d, '__getitem__') else torch.load(d)
        self.model.load_state_dict(d['model_state'], strict=resume)
        if resume:
            print('loading optimizer state, RNG state, step counts')
            print("""
            warning: optimizer lr, beta etc are restored with optimizer state,
            even if different values given on the command line, when resume=True
            """)
            self.opt.load_state_dict(d['optimizer_state'])
            self.exposure, self.iteration, self.epoch = d['step']
            self.set_random_state(d['random_state'])
            try:
                self.best_loss = d['best_loss']
                self.best_step = d['best_step']
            except KeyError:
                print('old checkpoint: no best_loss')
        else:
            print('fresh run transferring only model weights')

    def log(self, tag, d):
        # self.writer.add_scalars(tag, d, self.exposure)
        for k,v in d.items():
            self.writer.add_scalar(f'{tag}/{k}', v, self.exposure)
    
    def process_grad(self):
        r = {}
        if self.grad_clip is not None:
            r['grad_l2'] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
        return r

    def get_loss_components(self, result):
        # return {'error': result['mse']}
        return {'nll': result['nll']*self.nll_scale}

    def forward(self, batch):
        text_mask = batch['text_mask'].to(self.device, non_blocking=True)
        audio_mask = batch['audio_mask'].to(self.device, non_blocking=True)
        if self.model.text_encoder is not None:
            text = batch['text'].to(self.device, non_blocking=True)
        else:
            text = batch['text_emb']
            if text is None:
                raise ValueError("""
                    no text embeddings in dataset but no text encoder in model
                """)
            text = text.to(self.device, non_blocking=True)
        audio = batch['audio'].to(self.device, non_blocking=True)
        return self.model(text, audio, text_mask, audio_mask)
        # audio_mask TODO

    def _validate(self, valid_loader, ar_mask=None):
        """"""
        pops = defaultdict(list)
        self.model.eval()
        i = 0
        # for batch in tqdm(valid_loader, desc=f'validating epoch {self.epoch}'):
        vs = self.valid_size
        for batch in tqdm(
                it.islice(it.chain.from_iterable(it.repeat(valid_loader)), vs), 
                desc=f'validating epoch {self.epoch}', total=vs):
            with torch.no_grad():
                result = self.forward(batch)
                losses = {
                    k:v.item() 
                    for k,v in self.get_loss_components(result).items()}
                for k,v in losses.items():
                    pops[k].append(v)
                pops['loss'].append(sum(losses.values()))
            if i==0:
                self.rich_logs('valid', result)
                i+=1
        return {
            'logs':{k:np.mean(v) for k,v in pops.items()},
            'pops':pops
        }
    
    def compile(self):
        self.model = torch.compile(self.model, dynamic=True)

    def train(self):
        """Entry point to model training"""
        # self.save(self.model_dir / f'{self.epoch:04d}.ckpt')

        train_loader = DataLoader(
            self.train_dataset, self.batch_size,
            shuffle=not isinstance(self.train_dataset, torch.utils.data.IterableDataset),
            num_workers=self.n_jobs, pin_memory=self.gpu,
            collate_fn=self.train_dataset.dataset.collate_fn(),
            drop_last=True)

        valid_loader = DataLoader(
            self.valid_dataset, self.batch_size,
            shuffle=False, num_workers=self.n_jobs, pin_memory=self.gpu,
            collate_fn=self.valid_dataset.dataset.collate_fn())

        ##### validation loop
        def run_validation():
            logs = self._validate(valid_loader)['logs']
            self.log('valid', logs)
            if logs['loss'] < self.best_loss:
                self.best_loss = logs['loss']
                self.best_step = self.step
                self.save(self.model_dir / f'best.ckpt')

        try:
            epoch_size = self.epoch_size or len(train_loader)
        except TypeError:
            raise ValueError("specify epoch_size when using IterableDataset")

        # validate at initialization
        run_validation()

        while True:
            self.epoch += 1

            ##### training loop
            self.model.train()
            for batch in tqdm(
                # itertools incantation to support epoch_size larger than train set
                it.islice(
                    it.chain.from_iterable(it.repeat(train_loader)), epoch_size), 
                desc=f'training epoch {self.epoch}', total=epoch_size
                ):

                self.iteration += 1
                self.exposure += self.batch_size
                logs = {}

                ### forward+backward+optimizer step ###
                self.opt.zero_grad(set_to_none=True)
                result = self.forward(batch)
                losses = self.get_loss_components(result)
                loss = sum(losses.values())
                loss.backward()
                logs |= self.process_grad()
                self.opt.step()
                ########

                # log loss components
                logs |= {f'loss/{k}':v.item() for k,v in losses.items()}
                # log total loss
                logs |= {'loss':loss.item()}
                # log any other returned scalars
                logs |= {k:v.item() for k,v in result.items() if v.numel()==1}
                # other logs
                self.log('train', logs)

            if self.epoch%self.save_epochs == 0: 
                self.save(self.model_dir / f'{self.epoch:04d}.ckpt')
            self.save(self.model_dir / f'last.ckpt')
            run_validation()


    def add_audio(self, tag, audio):
        try:
            sr = self.rave_model.sampling_rate
        except Exception:
            sr = self.rave_model.sr
        self.writer.add_audio(
            tag, audio, 
            global_step=self.epoch, sample_rate=int(sr))

    def rich_logs(self, tag, result):
        """runs RAVE inference and logs audio"""
        z = result['predicted'].detach().cpu()
        gt = result['ground_truth'].detach().cpu()
        a = result['alignment'].detach().cpu()
        am = result['audio_mask'].detach().cpu()
        tm = result['text_mask'].detach().cpu()
        for i in range(3):
            nt = tm[i].sum()
            na = am[i].sum()
            z_i = z[i, :na].T[None]
            gt_i = gt[i, :na].T[None]
            with torch.inference_mode():
                audio_tf = self.rave_model.decode(z_i)[0]
                audio_gt = self.rave_model.decode(gt_i)[0]
            self.add_audio(f'{tag}/audio/tf/{i}', audio_tf)
            self.add_audio(f'{tag}/audio/gt/{i}', audio_gt)
            
            with torch.inference_mode():
                t = result['text'].detach()[i:i+1, :nt]
                z_ar,_ = self.model.inference(
                    t, stop=False, max_steps=int(na*1.2))
                z_ar = z_ar.cpu().transpose(1,2)
                z_ar_zero,_ = self.model.inference(
                    t, stop=False, max_steps=int(na*1.2), temperature=0.0)
                z_ar_zero = z_ar_zero.cpu().transpose(1,2)
                z_ar_half,_ = self.model.inference(
                    t, stop=False, max_steps=int(na*1.2), temperature=0.5)
                z_ar_half = z_ar_half.cpu().transpose(1,2)
                audio_ar = self.rave_model.decode(z_ar)[0]
                audio_ar_zero = self.rave_model.decode(z_ar_zero)[0]
                audio_ar_half = self.rave_model.decode(z_ar_half)[0]
            self.add_audio(f'{tag}/audio/ar/{i}', audio_ar)
            self.add_audio(f'{tag}/audio/ar/zerotemp/{i}', audio_ar_zero)
            self.add_audio(f'{tag}/audio/ar/halftemp/{i}', audio_ar_half)

            fig = plt.figure()
            a_i = a[i, :na, :nt].T
            plt.imshow(
                a_i, 
                interpolation='nearest', 
                aspect='auto', 
                origin='lower')
            self.writer.add_figure(f'{tag}/align/{i}', fig, global_step=self.epoch)

class Resumable:
    def __init__(self, checkpoint=None, resume=True, **kw):
        """
        Args:
            checkpoint: path to training checkpoint file
            resume: if True, retore optimizer states etc
                otherwise, restore only model weights (for transfer learning)
        """
        if checkpoint is not None:
            d = torch.load(checkpoint, map_location=torch.device('cpu'))
            print(f'loaded checkpoint {checkpoint}')
            # merges sub dicts, e.g. model hyperparameters
            deep_update(d['kw'], kw)
            self._trainer = Trainer(**d['kw'])
            self._trainer.load_state(d, resume=resume)
        else:
            self._trainer = Trainer(**kw)

    def train(self):
        # self._trainer.compile()
        self._trainer.train()

    # def test(self):
    #     self._trainer.test()

Resumable.__doc__ = Trainer.__init__.__doc__
Resumable.train.__doc__ = Trainer.train.__doc__

import torch._dynamo.config
torch._dynamo.config.verbose=True

if __name__=='__main__':
    # TODO: improve fire-generated help message
    # import pdb; pdb.runcall(fire.Fire, Resumable)

    try:
        fire.Fire(Resumable)
    except Exception as e:
        import traceback; traceback.print_exc()
        import pdb; pdb.post_mortem()