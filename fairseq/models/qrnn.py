# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/pytorch-qrnn')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fairseq import options, utils

from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model,
    register_model_architecture,
)

from torchqrnn import QRNN, QRNNLayer

class ForgetMult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        return torch.stack(result)


class QRNNLayer(nn.Module):
    """Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
        - all_hn (seq_len, batch, hidden_size): tensor containing the hidden state for all t
    """
    def __init__(self, input_size, hidden_size=None, window=1):
        super().__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.prevX = None

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
        Z, F, O = Y.chunk(3, dim=2)
        ###
        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        C = ForgetMult()(F, Z, hidden)

        # Apply (potentially optional) output gate
        H = torch.nn.functional.sigmoid(O) * C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        return H, C[-1:, :, :], C


class QRNN(torch.nn.Module):
    """Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (layers, batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (layers, batch, hidden_size): tensor containing the hidden state for t=seq_len
        - all_hn (seqlen, batch, hidden_size): tensor containing the hidden state for all tr in last layer
    """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True,
                 dropout=0, layers=None, **kwargs):
        super(QRNN, self).__init__()

        self.layers = torch.nn.ModuleList(layers if layers else [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.dropout = dropout

    def reset(self):
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []

        for i, layer in enumerate(self.layers):
            input, hn, all_hn = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden, all_hn


@register_model('qrnn')
class QrnnModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--window', type=int, metavar='N',
                            help='size of the convolutional window')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        encoder = QrnnEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            window=args.window,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
        )
        decoder = QrnnDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_output_units=encoder.output_units,
            window=args.window,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
        )
        return cls(encoder, decoder)


class QrnnEncoder(FairseqEncoder):
    """Qrnn encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1, window=1,
        dropout_in=0.1, dropout_out=0.1,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = dictionary.pad()
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)

        self.qrnn = QRNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            window=window,
        )
        for name, param in self.qrnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)

        self.output_units = hidden_size

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # apply QRNN
        self.qrnn.reset()
        x, h, _ = self.qrnn(x)
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        assert list(x.size()) == [seqlen, bsz, self.output_units]

        return {
            'encoder_out': (x, h),
            'encoder_padding_mask': None
        }

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number


class QrnnDecoder(FairseqIncrementalDecoder):
    """Qrnn decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, encoder_embed_dim=512, encoder_output_units=512, window=1,
        dropout_in=0.1, dropout_out=0.1,
    ):
        super().__init__(dictionary)
        self.hidden_size = hidden_size
        self.padding_idx = dictionary.pad()
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)

        self.encoder_output_units = encoder_output_units
        assert encoder_output_units == hidden_size, \
            'encoder_output_units ({}) != hidden_size ({})'.format(encoder_output_units, hidden_size)

        self.qrnn = QRNN(encoder_output_units, hidden_size, num_layers, window=window)
        for name, param in self.qrnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, = encoder_out[:2]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        pad_size = encoder_hiddens[0].size(0) - bsz
        x = self.padding(x, pad_size)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()

        self.qrnn.reset()
        outs, hidden, all_hidden = self.qrnn(x, encoder_hiddens)

        x = all_hidden
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # project back to size of vocabulary
        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        x = self.fc_out(x)

        return x[:bsz, :], attn_scores[:bsz, :]


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def padding(self, x, pad_size):
        return F.pad(x, (0, 0, 0, 0, 0, pad_size), 'constant', 0)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('qrnn', 'qrnn')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.window = getattr(args, 'window', args.window)


@register_model_architecture('qrnn', 'qrnn_iwslt_de_en')
def qrnn_wiseman_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    base_architecture(args)
