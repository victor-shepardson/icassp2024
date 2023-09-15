# adapted from coqui TTS
# Mozilla Public License

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

import numpy as np

from rtalign.text import TextEncoder


# TODO: options to convert pitch latent
# will try just training on hz first
def hz_to_z(hz):
    return torch.where(hz>0, hz.log2() - 7., torch.zeros_like(hz))
def z_to_hz(z):
    return 2**(z+7.)

# from scipy.stats import betabinom

# this implements the beta-binomial distribution PMF
# to eliminate the scipy dependency
def prior_coefs(prior_filter_len, alpha, beta):
    lg = torch.special.gammaln
    n = torch.tensor(prior_filter_len-1)
    k = torch.arange(prior_filter_len)
    a = torch.tensor(alpha)
    b = torch.tensor(beta)
    log_prior = (
        lg(n+1) - lg(k+1) - lg(n-k+1)
        + lg(k+a) + lg(n-k+b) - lg(n+a+b) 
        - lg(a) - lg(b) + lg(a+b)
    )
    prior = log_prior.exp()
    return prior.float().flip(0)[None,None]

@torch.jit.script
def do_prior_filter(attn_weight, filt, pad:int):
    prior_filter = F.conv1d(
        F.pad(attn_weight.unsqueeze(1), (pad, 0)), filt)
    return prior_filter.clamp_min(1e-6).log().squeeze(1)

@torch.jit.script
def addsoftmax(a, b):
    return (a + b).softmax(-1)

class MonotonicDynamicConvolutionAttention(nn.Module):
    """Dynamic convolution attention from
    https://arxiv.org/pdf/1910.10288.pdf

    This is an altered version where the static filter is replaced by the bias 
    of the linear layer leading to the dynamic filter

    original docstring follows:

    query -> linear -> tanh -> linear ->|
                                        |                                            mask values
                                        v                                              |    |
               atten_w(t-1) -|-> conv1d_dynamic -> linear -|-> tanh -> + -> softmax -> * -> * -> context
                             |-> conv1d_static  -> linear -|           |
                             |-> conv1d_prior   -> log ----------------|
    query: attention rnn output.
    Note:
        Dynamic convolution attention is an alternation of the location senstive attention with
    dynamically computed convolution filters from the previous attention scores and a set of
    constraints to keep the attention alignment diagonal.
        DCA is sensitive to mixed precision training and might cause instable training.
    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the value tensor.
        static_filter_dim (int): number of channels in the convolution layer computing the static filters.
        static_kernel_size (int): kernel size for the convolution layer computing the static filters.
        dynamic_filter_dim (int): number of channels in the convolution layer computing the dynamic filters.
        dynamic_kernel_size (int): kernel size for the convolution layer computing the dynamic filters.
        prior_filter_len (int, optional): [description]. Defaults to 11 from the paper.
        alpha (float, optional): [description]. Defaults to 0.1 from the paper.
        beta (float, optional): [description]. Defaults to 0.9 from the paper.
    """

    def __init__(
        self,
        query_dim,
        # embedding_dim,  # pylint: disable=unused-argument
        attention_dim,
        static_filter_dim=8, # unused
        static_kernel_size=21, # unused
        dynamic_filter_dim=8,
        dynamic_kernel_size=21,
        prior_filter_len=11,
        alpha=0.1,
        beta=0.9,
    ):
        super().__init__()
        self._mask_value = 1e-8
        self.dynamic_filter_dim = dynamic_filter_dim
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_filter_len = prior_filter_len
        self.attention_weights = None
        # setup key and query layers
        dynamic_weight_dim = dynamic_filter_dim * dynamic_kernel_size
        self.filter_mlp = torch.jit.script(nn.Sequential(
            nn.Linear(query_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, dynamic_weight_dim)#, bias=False)
        ))
        # self.query_layer = nn.Linear(query_dim, attention_dim)
        # self.key_layer = nn.Linear(
            # attention_dim, dynamic_filter_dim * dynamic_kernel_size, bias=False)
        # self.static_filter_conv = nn.Conv1d(
        #     1,
        #     static_filter_dim,
        #     static_kernel_size,
        #     padding=(static_kernel_size - 1) // 2,
        #     bias=False,
        # )
        # self.static_filter = nn.Parameter(torch.zeros(dynamic_filter_dim))
        # self.static_filter_layer = nn.Linear(
            # static_filter_dim, attention_dim, bias=False)
        # self.dynamic_filter_layer = nn.Linear(dynamic_filter_dim, attention_dim)
        self.post_mlp = torch.jit.script(nn.Sequential(
            nn.Linear(dynamic_filter_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        ))
        # self.v = nn.Linear(attention_dim, 1, bias=False)

        self.register_buffer("prior", prior_coefs(prior_filter_len, alpha, beta))
        # prior = betabinom.pmf(
            # range(prior_filter_len), prior_filter_len - 1, alpha, beta)
        # self.register_buffer("prior", torch.FloatTensor(prior).flip(0)[None,None])

    # pylint: disable=unused-argument
    def forward(self, query, inputs, mask):
        """
        query: [B, C_attn_rnn]
        inputs: [B, T_en, D_en]
        mask: [B, T_en]
        """
        B = query.shape[0]
        # compute prior filters
        # prior_filter = F.conv1d(F.pad(
        #     self.attention_weights.unsqueeze(1), (self.prior_filter_len - 1, 0)), 
        #     self.prior#.view(1, 1, -1)
        # )
        # prior_filter = torch.log(prior_filter.clamp_min_(1e-6)).squeeze(1)
        prior_filter = do_prior_filter(
            self.attention_weights, self.prior, self.prior_filter_len-1)

        G = self.filter_mlp(query)
        # compute dynamic filters
        pad = (self.dynamic_kernel_size - 1) // 2
        dynamic_filter = F.conv1d(
            self.attention_weights.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=pad,
            groups=B,
        )
        dynamic_filter = dynamic_filter.view(
            B, self.dynamic_filter_dim, -1).transpose(1, 2)

        # alignment = (
            # self.post_mlp(dynamic_filter).squeeze(-1)
            # + prior_filter
        # )
        # compute attention weights
        # attention_weights = F.softmax(alignment, dim=-1)
        attention_weights = addsoftmax(
            self.post_mlp(dynamic_filter).squeeze(-1), prior_filter)
        
        return self.update(attention_weights, inputs, mask)

    def update(self, attention_weights, inputs, mask=None):
        """this is split out to implement attention painting

        do weights really need to be masked here if it's post-softmax anyway?
        is it enough that the inputs are zero padded?
        """
        # apply masking
        if mask is not None:
            # attention_weights.masked_fill_(~mask, self._mask_value)
            attention_weights = attention_weights.masked_fill(
                ~mask, self._mask_value)
        self.attention_weights = attention_weights
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return context

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = inputs.new_zeros(B, T)
        self.attention_weights[:, 0] = 1.0

@torch.jit.script
def zoneout(x1:Tensor, x2:Tensor, p:float, training:bool):
    """
    Args:
        x1: old value
        x2: new value
        p: prob of keeping old value
        training: stochastic if True, expectation if False
    """
    keep = torch.full_like(x1, p)
    if training:
        keep = torch.bernoulli(keep)
    return torch.lerp(x2, x1, keep)

def apply_rnn(rnn, input, states, training, dropout_p=0.1, dropout_type=None):
    new_states = rnn(input, states)
    if dropout_type=='dropout':
        new_states = (
            F.dropout(s, dropout_p, training=training) 
            for s in new_states)
    elif dropout_type=='zoneout':
        new_states = (
            zoneout(s1, s2, dropout_p, training=training) 
            for s1,s2 in zip(states, new_states))
    else: assert dropout_type is None
    return new_states


log2pi = np.log(2*np.pi)
class DiagonalNormalMixture(nn.Module):
    def __init__(self, n:int=16):
        """n: number of mixture components"""
        super().__init__()
        self.n = n

    def n_params(self, size):
        """# of distribution parameters as a function of # latent variables"""
        return (2 * size + 1) * self.n

    def get_params(self, params:Tensor):
        """
        Args:
            params: Tensor[batch, time, n_params] 
        Returns:
            mu: Tensor[batch, time, n_latent, self.n]
            logsigma: Tensor[batch, time, n_latent, self.n]
            logitpi: Tensor[batch, time, self.n]
        """
        #means, log stddevs, logit weights
        locscale = params[...,:-self.n]
        logitpi = params[...,-self.n:]

        mu, logsigma = locscale.view(*params.shape[:-1], -1, self.n, 2).unbind(-1)
        return mu, logsigma, logitpi

    def forward(self, x:Tensor, params:Tensor):
        """mixture of diagonal normals negative log likelihood.
        should broadcast any number of leading dimensions
        Args:
            x: Tensor[batch, time, latent] or [..., latent]
            params: Tensor[batch, time, n_params] or [..., n_params]
        Return:
            negative log likelihood: Tensor[batch, time] or [...]
        """
        x = x[...,None] # broadcast against mixture component
        mu, logsigma, logitpi = self.get_params(params)
        logsigma = logsigma.clip(-7, 5) # TODO: clip range?
        # cmp_loglik = -0.5 * (
        #     ((x - mu) / logsigma.exp()) ** 2
        #     + log2pi
        # ) - logsigma
        # logpi = logitpi.log_softmax(-1)
        # return -(cmp_loglik.sum(-2) + logpi).logsumexp(-1)
        cmp_loglik = (
            (x - mu) # materialize huge tensor when calling from sample_n
            / logsigma.exp()) ** 2 

        cmp_loglik = -0.5 * (
            (cmp_loglik.sum(-2) + log2pi*x.shape[-2]) 
            ) - logsigma.sum(-2)

        logpi = logitpi.log_softmax(-1)
        return -(cmp_loglik + logpi).logsumexp(-1)
    
    def sample(self, params:Tensor, temperature:float=1.0, nsamp:int=None):
        if temperature != 1:
            # return self.sample_n(params, temperature, nsamp=128)
            if nsamp is not None:
                return self.sample_n(params, temperature, nsamp=nsamp)
            else:
                return self.sample_components(params, temperature)
        else:
            mu, logsigma, logitpi = self.get_params(params)
            idx = torch.distributions.Categorical(
                logits=logitpi).sample()[...,None,None].expand(*mu.shape[:-1],1)
            mu = mu.gather(-1, idx).squeeze(-1)
            logsigma = logsigma.gather(-1, idx).squeeze(-1)
            return mu + temperature*logsigma.exp()*torch.randn_like(logsigma)
    
    def sample_n(self, params:Tensor, temperature:float=1.0, nsamp=128):
        """
        draw nsamp samples,
        rerank and sample categorical with temperature
        Args:
            params: Tensor[..., n_params]    
        """
         # sample N 
        mu, logsigma, logitpi = self.get_params(params)
        logitpi = logitpi[...,None,:].expand(*logitpi.shape[:-1],nsamp,-1)
        # ..., nsamp, self.n
        # print(f'{logitpi.shape=}')
        idx = torch.distributions.Categorical(logits=logitpi).sample()
        # ..., nsamp
        # print(f'{idx.shape=}')
        idx = idx[...,None,:].expand(*mu.shape[:-1], -1)
        # ..., latent, nsamp 
        # print(f'{idx.shape=}')

        # mu is: ..., latent, self.n
        mu = mu.gather(-1, idx)
        logsigma = logsigma.gather(-1, idx)
        # ..., latent, nsamp
        # print(f'{mu.shape=}')

        samps = (mu + torch.randn_like(mu)*logsigma.exp()).moveaxis(-1, 0)
        # nsamp,...,latent
        # print(f'{samps.shape=}')

        # compute nll
        # here there is a extra first dimension (nsamp)
        # which broadcasts against the distribution params inside of self.forward,
        # to compute the nll for each sample
        nll = self(samps, params).moveaxis(0, -1)
        # ...,nsamp
        # print(f'{nll.shape=}')
        # print(f'{nll=}')

        # sample categorical with temperature
        idx = torch.distributions.Categorical(
            logits=-nll/(temperature+1e-5)).sample()
        # print(f'{idx.shape=}')
        # ...

        # select
        idx = idx[None,...,None].expand(1, *samps.shape[1:])
        # 1,...,latent
        # print(f'{idx.shape=}')
        samp = samps.gather(0, idx).squeeze(0)
        # ...,latent
        # print(f'{samp.shape=}')
        # print(f'{samp=}')
        return samp

    def sample_components(self, params:Tensor, temperature:float=1.0):
        """
        sample every mixture component with temperature,
        rerank and sample categorical with temperature.
        """
        # sample each component with temperature 
        mu, logsigma, _ = self.get_params(params)
        samps = mu + torch.randn_like(mu)*logsigma.exp()*temperature**0.5
        # ..., latent, self.n
        samps = samps.moveaxis(-1, 0)
        # self.n ...latent

        # compute nll for each sample
        # here there is a extra first dimension (nsamp)
        # which broadcasts against the distribution params inside of self.forward
        # to compute the nll for each sample
        nll = self(samps, params).moveaxis(0, -1)
        # ..., self.n

        # sample categorical with temperature
        if temperature > 1e-5:
            logits = nll.mul_(-1/temperature**0.5)
            idx = torch.distributions.Categorical(logits=logits).sample()
        else:
            idx = nll.argmin(-1)
        # print(f'{idx.shape=}')
        # ...

        # select
        idx = idx[None,...,None].expand(1, *samps.shape[1:])
        # 1,...,latent
        # print(f'{idx.shape=}')
        samp = samps.gather(0, idx).squeeze(0)
        # ...,latent
        # print(f'{samp.shape=}')
        # print(f'{samp=}')
        return samp

    def metrics(self, params:Tensor):
        mu, logsigma, logitpi = self.get_params(params)
        ent = torch.distributions.Categorical(logits=logitpi).entropy().detach()
        return {
            'logsigma_min': logsigma.min().detach(),
            'logsigma_median': logsigma.median().detach(),
            'logsigma_max': logsigma.max().detach(),
            'pi_entropy_min': ent.min(),
            'pi_entropy_median': ent.median(),
            'pi_entropy_max': ent.max(),
        } 

class StandardNormal(nn.Module):
    def __init__(self):
        super().__init__()

    def n_params(self, size):
        return size

    def forward(self, x:Tensor, params:Tensor):
        """standard normal negative log likelihood
        Args:
            x: Tensor[batch, time, channel]
            params: Tensor[batch, time, n_params]    
        Return:
            likelihood: Tensor[batch, time]
        """
        mu = params
        loglik = 0.5 * (
            (x - mu) ** 2 + log2pi
        )
        return loglik.sum(-1)
    
    def sample(self, params:Tensor, temperature:float=1.0):
        return params + temperature*torch.randn_like(params)
    
    def metrics(self, params:Tensor):
        return {}

class DiagonalNormal(nn.Module):
    def __init__(self):
        super().__init__()

    def n_params(self, size):
        return size * 2

    def forward(self, x:Tensor, params:Tensor):
        """diagonal normal negative log likelihood
        Args:
            x: Tensor[batch, time, channel]
            params: Tensor[batch, time, n_params]    
        Return:
            likelihood: Tensor[batch, time]
        """
        mu, logsigma = params.chunk(2, -1)
        logsigma = logsigma.clip(-7, 5) # TODO: clip range
        loglik = 0.5 * (
            ((x - mu) / logsigma.exp()) ** 2
            + log2pi
        ) + logsigma
        return loglik.sum(-1)
    
    def sample(self, params:Tensor, temperature:float=1.0):
        mu, logsigma = params.chunk(2, -1)
        return mu + temperature*logsigma.exp()*torch.randn_like(logsigma)
    
    def metrics(self, params:Tensor):
        mu, logsigma = params.chunk(2, -1)
        return {
            'logsigma_min': logsigma.min().detach(),
            'logsigma_median': logsigma.median().detach(),
            'logsigma_max': logsigma.max().detach(),
        }

# adapted from https://github.com/NVIDIA/tacotron2/
class TacotronDecoder(nn.Module):
    """Tacotron2 decoder. We don't use Zoneout but Dropout between RNN layers.
    Args:
        in_channels (int): number of input channels.
        frame_channels (int): number of feature frame channels.
        prenet_type (string): 'original' or 'bn'.
        prenet_dropout (float): prenet dropout rate.
        separate_stopnet (bool): if true, detach stopnet input to prevent gradient flow.
        max_decoder_steps (int): Maximum number of steps allowed for the decoder. Defaults to 10000.
        text_encoder: dict of text encoder kwargs
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init
    def __init__(
        self,
        in_channels=768, # text embedding dim
        frame_channels=16, # RAVE latent dim
        dropout=0.1,
        likelihood_type='diagonal',#'normal'#'mixture'
        mixture_n=16,
        dropout_type='dropout', #'zoneout'
        prenet_type='original', # disabled
        prenet_dropout=0.5,
        prenet_layers=2,
        prenet_size=256,
        separate_stopnet=True, # disabled
        max_decoder_steps=10000,
        text_encoder:Dict=None,
        rnn_size=1024,
        rnn_bias=True,
        decoder_layers=1,
        decoder_size=None,
        learn_go_frame=False,
        pitch_xform=False,
    ):
        super().__init__()
        self.frame_channels = frame_channels
        self.pitch_xform = pitch_xform
        # self.r_init = r
        # self.r = r
        self.encoder_embedding_dim = in_channels
        self.separate_stopnet = separate_stopnet
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = 0.5

        if text_encoder is not None:
            self.text_encoder = TextEncoder(**text_encoder)
        else:
            self.text_encoder = None

        decoder_size = decoder_size or rnn_size

        # model dimensions
        self.decoder_layers = decoder_layers
        self.query_dim = rnn_size
        self.decoder_rnn_dim = decoder_size
        self.prenet_dim = prenet_size
        self.attn_dim = 128
        self.p_attention_dropout = dropout
        self.p_decoder_dropout = dropout
        self.dropout_type = dropout_type

        # memory -> |Prenet| -> processed_memory
        prenet_dim = self.frame_channels
        self.prenet_dropout = prenet_dropout
        self.prenet = Prenet(
            prenet_dim, prenet_type, 
            out_features=[self.prenet_dim]*prenet_layers, 
            bias=False
        )

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + in_channels, self.query_dim, bias=rnn_bias)

        # self.attention = init_attn(
            # attn_type=attn_type,
        self.attention = MonotonicDynamicConvolutionAttention(
            query_dim=self.query_dim,
            # embedding_dim=in_channels,
            attention_dim=128,
        )

        self.decoder_rnn = nn.LSTM(
            self.query_dim + in_channels, self.decoder_rnn_dim, 
            num_layers=self.decoder_layers,
            bias=rnn_bias, batch_first=True, 
            dropout=0 if decoder_layers==1 else dropout)
        
        if likelihood_type=='normal':
            self.likelihood = StandardNormal()
        elif likelihood_type=='diagonal':
            self.likelihood = DiagonalNormal()
        elif likelihood_type=='mixture':
            self.likelihood = DiagonalNormalMixture(mixture_n)
        else:
            raise ValueError(likelihood_type)

        self.linear_projection = nn.Linear(
            # self.decoder_rnn_dim + in_channels, self.frame_channels * self.r_init)
            self.decoder_rnn_dim + in_channels, 
            self.likelihood.n_params(self.frame_channels))

        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(
                # self.decoder_rnn_dim + self.frame_channels * self.r_init, 1, 
                self.decoder_rnn_dim + self.frame_channels, 1, 
                # bias=True, init_gain="sigmoid"
                ),
        )
        self.memory = None
        # self.memory_truncated = None

        self.learn_go_frame = learn_go_frame
        if learn_go_frame:
            self.go_frame = nn.Parameter(
                torch.zeros(1, self.frame_channels))
            self.go_query = nn.Parameter(
                torch.zeros(1, self.query_dim))
            self.go_attention_rnn_cell_state = nn.Parameter(
                torch.zeros(1, self.query_dim))
            self.go_decoder_hidden = nn.Parameter(
                torch.zeros(self.decoder_layers, 1, self.decoder_rnn_dim))
            self.go_decoder_cell = nn.Parameter(
                torch.zeros(self.decoder_layers, 1, self.decoder_rnn_dim))
            self.go_context = nn.Parameter(
                torch.zeros(1, self.encoder_embedding_dim))

    def get_go_frame(self, inputs):
        B = inputs.size(0)

        if hasattr(self, 'go_frame'):
            return self.go_frame.expand(B,-1)
        else:
            return inputs.new_zeros(B, self.frame_channels)

    def _init_states(self, inputs, mask, keep_states=False):
        B = inputs.size(0)
        if not keep_states:
            if self.learn_go_frame:
                self.query = self.go_query.expand(B, -1)
                self.attention_rnn_cell_state = self.go_attention_rnn_cell_state.expand(B, -1)
                self.decoder_hidden = self.go_decoder_hidden.expand(-1, B, -1)
                self.decoder_cell = self.go_decoder_cell.expand(-1, B, -1)
                self.context = self.go_context.expand(B, -1)
            else:
                self.query = inputs.new_zeros(B, self.query_dim)
                self.attention_rnn_cell_state = torch.zeros_like(self.query)
                self.decoder_hidden = inputs.new_zeros(
                    self.decoder_layers, B, self.decoder_rnn_dim)
                self.decoder_cell = torch.zeros_like(self.decoder_hidden)
                self.context = inputs.new_zeros(B, self.encoder_embedding_dim)
        self.inputs = inputs
        self.mask = mask

        self.attention.init_states(inputs)

    def decode_core(self, memory, alignment=None):
        """run step of attention loop
        Args:
            memory: B x D_audio
        """
        query_input = torch.cat((memory, self.context), -1)
        # self.query and self.attention_rnn_cell_state : B x D_attn_rnn
        self.query, self.attention_rnn_cell_state = apply_rnn(self.attention_rnn,
            query_input, (self.query, self.attention_rnn_cell_state),
            self.training, 
            self.p_attention_dropout, dropout_type=self.dropout_type)
        # B x D_en
        if alignment is None:
            # compute attention
            self.context = self.attention(self.query, self.inputs, self.mask)
        else:
            # set attention
            self.context = self.attention.update(alignment, self.inputs, self.mask)

        return self.query, self.context, self.attention.attention_weights

    def decode_post(self, hidden, context, lengths=None):
        """run post-decoder (step or full time dimension)
        
        Args:
            hidden: B x T_audio x channel (hidden state after attention net)
            context: B x T_audio x D_text (audio-aligned text features)
        Returns:
            hidden: B x T_audio x channel (hidden state after decoder net)
            output_params: B x T_audio x channel (likelihood parameters)
        """
        decoder_rnn_input = torch.cat((hidden, context), -1)
        if lengths is not None:
            decoder_rnn_input = nn.utils.rnn.pack_padded_sequence(
                decoder_rnn_input, lengths, batch_first=True, enforce_sorted=False)

        # self.decoder_hidden and self.decoder_cell: B x D_decoder_rnn
        hidden, (self.decoder_hidden, self.decoder_cell) = self.decoder_rnn( 
            decoder_rnn_input, (self.decoder_hidden, self.decoder_cell))
        
        if lengths is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(
                hidden, batch_first=True)
        # # B x T x (D_decoder_rnn + D_text)
        decoder_hidden_context = torch.cat((hidden, context), dim=-1)
        # B x T x self.frame_channels
        output_params = self.linear_projection(decoder_hidden_context)
        return hidden, output_params

    def predict_stop(self, decoder_state, output):
        # B x (D_decoder_rnn + (self.r * self.frame_channels))
        stopnet_input = torch.cat((decoder_state, output), dim=-1)
        if self.separate_stopnet:
            stopnet_input = stopnet_input.detach()
        stop_token = self.stopnet(stopnet_input)
        return stop_token
    
    def latent_map(self, z):
        if self.pitch_xform:
            z = torch.cat((
                hz_to_z(z[...,:1]),
                z[...,1:]
            ), -1) 
        return z

    def latent_unmap(self, z):
        if self.pitch_xform:
            z = torch.cat((
                z_to_hz(z[...,:1]),
                z[...,1:]
            ), -1) 
        return z

    def forward(self, inputs, audio, mask, audio_mask, prenet_dropout=None):
        r"""Train Decoder with teacher forcing.
        Args:
            inputs: raw or encoded text.
            audio: audio frames for teacher-forcing.
            mask: text mask for sequence padding.
            audio_mask: audio mask for loss computation.
            prenet_dropout: if not None, override original value
                (to implement e.g. annealing)
        Shapes:
            - inputs: 
                FloatTensor (B, T_text, D_text)
                or LongTensor (B, T_text)
            - audio: (B, T_audio, D_audio)
            - mask: (B, T_text)
            - audio_mask: (B, T_audio)
            - stop_target TODO

            - outputs: (B, T_audio, D_audio)
            - alignments: (B, T_audio, T_text)
            - stop_tokens: (B, T_audio)

        """
        audio_lengths = audio_mask.sum(-1).cpu()
        ground_truth = audio

        # print(f'{audio[...,0].min()=}')
        audio = self.latent_map(audio)
        # print(f'{audio[...,0].min()=}')

        memory = self.get_go_frame(inputs).unsqueeze(0)
        # memories = self._reshape_memory(audio)
        memories = audio.transpose(0, 1)
        memories = torch.cat((memory, memories), dim=0)
        memories = self.prenet(memories, dropout=self.prenet_dropout)

        if inputs.dtype==torch.long:
            assert self.text_encoder is not None
            assert inputs.ndim==2
            inputs = self.text_encoder(inputs).hidden_states[-1]

        self._init_states(inputs, mask=mask)
        # self.attention.init_states(inputs)

        hidden, contexts, alignments = [], [], []
        for memory in memories[:-1]:
            h, context, alignment = self.decode_core(memory)
            hidden.append(h)
            contexts.append(context)
            alignments.append(alignment)
        hidden = torch.stack(hidden, 1)
        contexts = torch.stack(contexts, 1)
        alignments = torch.stack(alignments, 1)

        hidden, output_params = self.decode_post(
            hidden, contexts, audio_lengths)

        # alignments = self._parse_outputs(alignments)

        m = audio_mask[...,None]
        nll = self.likelihood(audio*m, output_params*m)
        nll = nll.masked_select(audio_mask).mean()

        with torch.inference_mode():
            # outputs = self.likelihood.sample(output_params, temperature=0)
            outputs = self.likelihood.sample(output_params, temperature=1) # low memory

        d = audio*m - outputs*m
        error = (d*d).mean() * m.numel() / m.float().sum()

        stop_loss = None
        # TODO
        # stop_tokens = self.predict_stop(hidden, outputs)
        # stop_loss = compute_stop_loss(stop_target, stop_tokens)

        outputs = self.latent_unmap(outputs)

        return {
            'text': inputs,
            'mse': error,
            'nll': nll,
            # 'stop_loss': stop_loss,
            'predicted': outputs,
            'ground_truth': ground_truth,
            'alignment': alignments,
            # 'stop': stop_tokens,
            'audio_mask': audio_mask,
            'text_mask': mask,
            **self.likelihood.metrics(output_params)
        }
    
    def reset(self, inputs):
        r"""
        reset before using `step`

        Args:
            inputs: (B, T_text, D_text)

        """
        assert inputs.ndim==3, f'{inputs.shape=}'

        self.memory = self.get_go_frame(inputs) # B x D_audio
        self._init_states(inputs, mask=None, keep_states=False)
        # self.attention.init_states(inputs)

    # TODO: update backend.py / inference to use `debug=True` version, remove flag
    def step(self, alignment=None, audio_frame=None, temperature=1.0, debug=False):
        r"""
        single step of inference.

        optionally supply `alignment` to force the alignments.
        optionally supply `audio_frame` to set the previous frame.

        Args:
            alignment: B x T_text
            audio_frame: B x D_audio
        Returns:
            output: B x D_audio
            alignment: B x T_text
            stop_token
        """
        if alignment is not None:
            assert alignment.ndim==2, f'{alignment.shape=}'

        if audio_frame is not None:
            self.memory[:] = self.latent_map(audio_frame)

        memory = self.prenet(self.memory)
        hidden, context, alignment = self.decode_core(memory, alignment=alignment)
        hidden, output_params = self.decode_post(hidden[:,None], context[:,None])
        decoder_output = self.likelihood.sample(
            output_params, temperature=temperature).squeeze(1)

        self.memory[:] = decoder_output

        decoder_output = self.latent_unmap(decoder_output)

        if debug:
            return dict(
                output=decoder_output,
                alignment=alignment,
                stop_prob=0,
                params=output_params
            )
        else: 
            return decoder_output, alignment, 0

    def inference(self, inputs, stop=True, max_steps=None, temperature=1.0):
        r"""Decoder inference without teacher forcing and use
        Stopnet to stop decoder.
        Args:
            inputs: Text Encoder outputs.
            stop: use stop gate
            max_steps: stop after this many decoder steps
            temperature: for the sampling distribution
        Shapes:
            - inputs: (B, T_text, D_text)
            - outputs: (B, T_audio, D_audio)
            - alignments: (B, T_text, T_audio)
            - stop_tokens: (B, T_audio)
        """
        max_steps = max_steps or self.max_decoder_steps

        self.reset(inputs)

        outputs, alignments = [], []
        while True:
            output, alignment, stop_token = self.step(temperature=temperature)
            outputs.append(output)
            alignments.append(alignment)
            # memory = self.prenet(memory)
            # h, context, attention_weights = self.decode_core(memory)

            # # output_params, alignment = self.decode(memory)
            # decoder_output = self.likelihood.sample(output_params, temperature=temperature)
            # stop_token = self.predict_stop(self.decoder_hidden, decoder_output)
            # stop_token = torch.sigmoid(stop_token.data)
            # outputs += [decoder_output.squeeze(1)] # squeeze?
            # stop_tokens += [stop_token.sigmoid()]
            # alignments += [alignment]

            if stop and stop_token>self.stop_threshold:
                break
            if len(outputs) == max_steps:
                if stop:
                    print(f"   > Decoder stopped with `max_decoder_steps` {self.max_decoder_steps}")
                break

            # memory = self._update_memory(decoder_output)
            # memory = decoder_output
            # t += 1

        outputs = torch.stack(outputs, 1)
        alignments = torch.stack(alignments, 1)

        # outputs, stop_tokens, alignments = self._parse_outputs(
            # outputs, stop_tokens, alignments)

        return outputs, alignments
    

class Prenet(nn.Module):
    """Tacotron specific Prenet with an optional Batch Normalization.
    Note:
        Prenet with BN improves the model performance significantly especially
    if it is enabled after learning a diagonal attention alignment with the original
    prenet. However, if the target dataset is high quality then it also works from
    the start. It is also suggested to disable dropout if BN is in use.
        prenet_type == "original"
            x -> [linear -> ReLU -> Dropout]xN -> o
        prenet_type == "bn"
            x -> [linear -> BN -> ReLU -> Dropout]xN -> o
    Args:
        in_features (int): number of channels in the input tensor and the inner layers.
        prenet_type (str, optional): prenet type "original" or "bn". Defaults to "original".
        prenet_dropout (bool, optional): dropout rate. Defaults to True.
        dropout_at_inference (bool, optional): use dropout at inference. It leads to a better quality for some models.
        out_features (list, optional): List of output channels for each prenet block.
            It also defines number of the prenet blocks based on the length of argument list.
            Defaults to [256, 256].
        bias (bool, optional): enable/disable bias in prenet linear layers. Defaults to True.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        in_features,
        prenet_type="original",
        dropout_at_inference=False,
        out_features=[256, 256],
        bias=True,
    ):
        super().__init__()
        self.prenet_type = prenet_type
        self.dropout_at_inference = dropout_at_inference
        in_features = [in_features] + out_features[:-1]
        # if prenet_type == "bn":
        #     self.linear_layers = nn.ModuleList(
        #         [LinearBN(in_size, out_size, bias=bias) for (in_size, out_size) in zip(in_features, out_features)]
        #     )
        # elif prenet_type == "original":
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_size, out_size, bias=bias)
            for (in_size, out_size) in zip(in_features, out_features)
        ])

    def forward(self, x, dropout=0.5):
        for linear in self.linear_layers:
            if dropout:
                x = F.dropout(
                    F.relu(linear(x)), p=dropout, 
                    training=self.training or self.dropout_at_inference)
            else:
                x = F.relu(linear(x))
        return x