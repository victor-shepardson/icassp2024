from typing import Optional, Callable, Union
from numbers import Number

from threading import Thread, RLock
from queue import Queue
import time
from fractions import Fraction

import fire

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

import sounddevice as sd

from rtalign import TacotronDecoder, TextEncoder

class Clock:
    """simple clock which can sleep until the next tick.
    does not drift, but does jitter
    """
    def __init__(self, tick_seconds:Number):
        """
        Args:
            tick_seconds: tick length in seconds.
                if this is a Fraction, there should be no drift
        """
        self.tick_ns = 1_000_000_000*tick_seconds
        self.reset()

    def reset(self):
        self.first_tick_ns = None
        self.ticks = 0

    def tick(self):
        """sleep until the next tick and increment tick count"""
        if self.first_tick_ns is None:
            self.first_tick_ns = time.perf_counter_ns()
            time.sleep(0)
            self.ticks = 1
        else:
            next_tick = self.ticks + 1
            now = time.perf_counter_ns()
            ns_elapsed = now - self.first_tick_ns
            delay_ns = (next_tick-1) * self.tick_ns - ns_elapsed
            # print(delay_ns)
            if delay_ns < 0:
                time.sleep(0)
                delay_ms = delay_ns*1e-6
                print(f'rtalign: Clock: late {-delay_ms:0.3f} milliseconds')
            else:
                time.sleep(delay_ns*1e-9)
            self.ticks = next_tick


class Backend:
    def __init__(self,
        checkpoint:str,
        clock_tick_seconds=Fraction(2048,48000),
        rave_path:str=None,
        audio_in:str=None,
        audio_out:Union[str,int]=None,
        audio_block:int=64
        ):
        """
        Args:
            checkpoint: path to rtalign checkpoint file
            clock_tick_seconds: length of a vocoder frame
                (inferred if rave_path is supplied)
            rave_path: path to rave vocoder to use python sound
            audio_in: sounddevice input if using python audio
            audio_out: sounddevice output if using python audio
            audio_block: block size if using python audio
        """
        self.playing = False

        # self.temperature = 0.9
        self.temperature = 1.

        # If set in client code, will be read as a one-time alignment for the next inference step and then reset to None
        self.target_alignment = None

        ckpt = torch.load(checkpoint, map_location='cpu')
        self.model = TacotronDecoder(**ckpt['kw']['model'])
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        # print(f'{self.model.frame_channels=}')
        self.use_pitch = hasattr(self.model, 'pitch_xform') and self.model.pitch_xform

        ### synthesis in python:
        ### sounddevice callback-driven
        if rave_path is not None:
            self.rave = torch.jit.load(rave_path, map_location='cpu')
            self.rave.eval()
            self.block_size = int(self.rave.decode_params[1])
            try:
                sr = int(self.rave.sampling_rate)
            except Exception:
                sr = int(self.rave.sr)
            with torch.inference_mode():
                # warmup
                if hasattr(self.rave, 'full_latent_size'):
                    latent_size = self.rave.latent_size + int(
                        hasattr(self.rave, 'pitch_encoder'))
                else:
                    latent_size = self.rave.cropped_latent_size
                self.rave.decode(torch.zeros(1,latent_size,1))


            self.active_frame = None # tensor
            self.future_frame = None # thread
            self.frame_counter = 0
            sd.default.device = audio_out
            sd.default.samplerate = sr # could cause an error if device uses a different sr

            print(f"RAVE SAMPLING RATE: {sr}")
            print(f"DEVICE SAMPLING RATE: {sd.default.samplerate}")
            devicelist = sd.query_devices()
            print(devicelist)

            for dev in devicelist:
                print(f"{dev['index']}: '{dev['name']}' {dev['hostapi']} (I/O {dev['max_input_channels']}/{dev['max_input_channels']}) (SR: {dev['default_samplerate']})")


            self.stream = sd.Stream(
                callback=self.audio_callback,
                samplerate=sr, 
                blocksize=audio_block, 
                #device=(audio_in, audio_out)
                device=audio_out,
            )
            
            assert self.stream.samplerate == sr, f"""
                failed to set sample rate to {sr} from sounddevice
                """
            
            self.clock = None
            self.loop_thread = None
        ### synthesis in supercollider:
        ### clocked loop-driven
        else:
            self.rave = None
            self.stream = None
            self.clock = Clock(tick_seconds=clock_tick_seconds)
            self.loop_thread = Thread(target=self.loop, daemon=True)
        
        if self.model.text_encoder is None:
            self.text_model = TextEncoder()
        else:
            self.text_model = self.model.text_encoder

        self.text = None
        self.text_t = None
        self.align_t = None

        self.mode = 'infer'
        self.one_shot_paint = False
        self.latent_feedback = False

        self.lock = RLock()
        self.text_update_thread = None

        self.frontend_q = Queue()
        self.synth_q = Queue()
        
        self.frame_callback = None

        self.state = None

        self.needs_reset = False

    @property
    def num_latents(self):
        return self.model.frame_channels

    def start(self):
        """
        start autoregressive alignment & latent frame generation
        """
        self.playing = True
        if self.rave is None:
            if not self.loop_thread.is_alive():
                self.run_thread = True
                self.loop_thread.start()
        else:
            if not self.stream.active:
                self.stream.start()

    def pause(self):
        """
        pause autoregressive generation
        """
        self.playing = False
        if self.clock is not None:
            self.clock.reset()

    def reset(self):
        """
        reset the model state and alignments history
        """
        self.needs_reset = True

    def cleanup(self):
        """
        Cleanup any resources
        """
        self.playing = False
        self.run_thread = False

    def set_text(self, text:str) -> int:
        """
        Compute embeddings for & store a new text, replacing the old text.
        Returns the number of embedding tokens

        Args:
            text: input text as a string

        Returns:
            length of text in tokens
        """
        self.text = text
        tt = self.text_model.text_bytes(self.text)
        # pad too-short inputs
        tt = self.text_model.pad(tt)

        self.text_tokens = tt

        if (
            self.text_update_thread is not None 
            and self.text_update_thread.is_alive()
        ):
            print('warning: text update still pending')
        self.text_update_thread = Thread(target=self._update_text, daemon=True)
        self.text_update_thread.start()

        return self.text_tokens.shape[1]

    def _update_text(self):
        # lock should keep multiple threads from trying to run the text encoder
        # at once in case `input_text`` is called rapidly
        with self.lock:
            self.text_t = self.text_model.encode(self.text_tokens)
            self.reset()

    def set_alignment(self, align_t:Union[Tensor, npt.ArrayLike]):
        """
        Send an alignment frame to the backend. If mode==paint this frame will
        be painted directly to the alignments, if mode==infer this method is ignored.

        Args:
            align_t: [batch, text length in tokens]
        """
        # attribute assignment is atomic
        if type(align_t) == np.ndarray:
            align_t = torch.from_numpy(align_t)
        self.align_t = align_t.clone() # prevent frontend from changing this tensor

    def set_mode(self, mode:str):
        """
        Set alignment generation mode: <infer|paint>
        """
        self.mode = mode
    
    def set_one_shot_paint(self, val:bool=True) -> None:
        """
        Paint mode will be forced on the next frame only, reading the current alignment paint value.
        """
        self.one_shot_paint = val

    def set_latent_feedback(self, b:bool):
        self.latent_feedback = b

    def process_frame(self, 
            mode:str, 
            audio_t:Optional[Tensor]=None, 
            align_t:Optional[Tensor]=None
        ) -> tuple[Tensor, Tensor]:
        """
        Generate an audio(rave latents) and alignment frame.
        Note: if self.target_alignment is set, it will override align_t for the current frame only.

        Args:
            mode: 
                'paint' to use align_t for alignments
                'infer' to use learned attention mechanism
            audio_t: [batch, RAVE latent size]
                last frame of audio feature if using `latent feedback` mode
            align_t: [batch, text length in tokens]
                explicit alignments if using `paint` mode
        Returns:
            audio_t: [batch, RAVE latent size]
                next frame of audio feature
            align_t: [batch, text length in tokens]
                alignments to text
        """

        if self.one_shot_paint:
            mode='paint'

        with torch.inference_mode():
            # if mode=='paint':
            audio_t, align_t, _ = self.model.step(
                alignment=align_t, audio_frame=audio_t, 
                temperature=self.temperature)
            # elif mode=='infer':
            #     audio_t, align_t, _ = self.model.step(
            #         alignment=None, audio_frame=audio_t,
            #         temperature=self.temperature)
            # else:
                # raise ValueError

        return audio_t, align_t
    
    def audio_callback(self,
            indata: np.ndarray, outdata: np.ndarray, #[frames x channels]
            frames: int, time, status):
        """sounddevice callback main loop"""

        for i in range(outdata.shape[0]):
            # if the current frame is exhausted, delete it
            if self.active_frame is not None:
                if self.frame_counter == self.active_frame.shape[0]:
                    self.active_frame = None
                    self.frame_counter = 0

            # if no active frame, try to join thread
            if self.active_frame is None:
                if self.future_frame is not None:
                    self.future_frame.join()
                    self.future_frame = None
                    self.active_frame = self.synth_q.get_nowait()

            # if no future frame, start thread
            if self.playing and self.future_frame is None: 
                # use ADC input time as timestamp
                timestamp = time.inputBufferAdcTime
                self.future_frame = Thread(
                    target=self.step, args=(timestamp,), daemon=True)
                self.future_frame.start()

            if self.active_frame is None:
                outdata[:,:] = 0
                if self.playing:
                    print(f'audio: dropped frame')
                return
            else:
                # read next audio sample out of active model frame 
                outdata[i,:] = self.active_frame[self.frame_counter]
                self.frame_counter += 1

    
    def step(self, timestamp):

        if self.frontend_q.qsize() > 100:
            self.frontend_q.get()
            print('dropping frame: frontend queue full')

        is_reset = False
        if self.needs_reset:
            if self.text_t is not None:
                self.model.reset(self.text_t)
                self.needs_reset = False
                is_reset = True

        if self.model.memory is None:
            print('skipping: model not initialized')
            if self.rave is not None:
                self.synth_q.put(None)
        else:
            # with self.lock:

            if self.state is not None and self.latent_feedback:
                # pass the last vocoder frame back in
                audio_t = self.state['audio_t']
            else:
                audio_t = None

            if self.mode == 'paint':
                align_t = self.align_t
            else:
                align_t = None

            ##### run the TTS model
            audio_t, align_t = self.process_frame(
                self.mode, audio_t=audio_t, align_t=align_t)
            #####

            self.state = {
                'reset':bool(is_reset),
                'audio_t':audio_t, 
                'align_t':align_t,
                'timestamp':timestamp
                }
            # frontent-supplied callback
            if self.frame_callback is not None:
                # callbacks may modify state in-place
                self.frame_callback(self.state)
            # queue for sending to frontend
            self.frontend_q.put(self.state)

            if self.rave is not None:
                with torch.inference_mode():
                    ### run the vocoder
                    audio = self.rave.decode(self.state['audio_t'][...,None])[0].T
                    # time, channel
                    ###
                self.synth_q.put(audio)
    
    def loop(self):
        """
        clocked main loop
        """
        while self.run_thread:
            if self.playing:
                self.clock.tick()
                # use tick time as timestamp
                self.step(time.time())
            else:
                time.sleep(1e-2)


def main(checkpoint='../rtalign_004_0100.ckpt'):
    from rtalign.gui.senders import DirectOSCSender
    b = Backend(checkpoint, callback=DirectOSCSender())
    b.start()
    print('setting text')
    n = b.set_text('hey, hi, hello, test. text. this is a text.') # returns the token length of the text
    print(f'text length is {n} tokens')
    print('setting alignment')
    i = 0
    while True:
        b.set_alignment(torch.randn(1,n).softmax(-1)) # ignored if mode == paint
        if i==300:
            print('setting text')
            n = b.set_text('this is a new text')
        if i==500:
            print('setting mode')
            b.set_mode('paint') # the other available mode is 'infer'

        while not b.q.empty():
            r = b.q.get()
            a = r['align_t'].argmax().item()
            print('█'*(a+1) + ' '*(n-a-1) + '▏')

            print(' '.join(f'{x.item():+0.1f}' for x in r['audio_t'][0]))
            
        time.sleep(1e-2)
        i += 1

if __name__=='__main__':
    fire.Fire(main)
