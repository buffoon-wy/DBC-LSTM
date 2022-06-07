# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        out_dims = 100  # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 56  # 3 cameras
        # out_dims = 100  # 5 cameras
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class LstmEncoder(nn.Module):
    def __init__(self, encoder_dim, act_dim, lstm_hid_sizes=128):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.act_dim = act_dim

        self.fc1 = nn.Linear((self.encoder_dim+self.act_dim), lstm_hid_sizes)
        self.lstm_module = nn.LSTM(lstm_hid_sizes, lstm_hid_sizes, batch_first=True)
        self.fc2 = nn.Linear(lstm_hid_sizes, encoder_dim)

    def forward(self, hist_obs_after_encoder, hist_act, hist_seg_len):

        x = torch.cat([hist_obs_after_encoder, hist_act], dim=-1)

        x = self.fc1(x)
        x = torch.relu(x)
        x, (lstm_hidden_state, lstm_cell_state) = self.lstm_module(x)
        x = self.fc2(x)
        hist_out = torch.tanh(x)

        #   History output mask to reduce disturbance cased by none history memory
        hist_index = (hist_seg_len - 1).view(-1, 1).repeat(1, self.encoder_dim).unsqueeze(1).long()
        extracted_memory = torch.gather(hist_out, 1, hist_index).squeeze(1)

        # the feather extracted from the history observations and actions
        return extracted_memory


class PixelLstmEncoderCarla096(nn.Module):
    """
    Convolutional encoder of pixels LSTM observations.
    # added by 51
    """
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, stride, device, action_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.device = device
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.outputs = dict()

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 32, 3, stride=2))
        self.convs.append(nn.Conv2d(32, 32, 3, stride=2))
        self.convs.append(nn.Conv2d(32, 32, 3, stride=2))
        self.convs.append(nn.Conv2d(32, 32, 3, stride=2))

        # out_dims = 56  # 3 cameras
        out_dims = 100  # 5 cameras
        self.fc1 = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.lstm_encoder = LstmEncoder(self.feature_dim, action_shape[0])
        self.fc2 = nn.Linear(2 * self.feature_dim, self.feature_dim)

    def forward_conv(self, obs, detach):
        obs = obs / 255.
        self.outputs['obs'] = obs
        conv = self.convs[0](obs)
        conv = torch.relu(conv)
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.contiguous().view(conv.size(0), -1)

        if detach:
            h = h.detach()

        h_fc1 = self.fc1(h)
        self.outputs['fc1'] = h_fc1

        conv_out = self.ln(h_fc1)
        self.outputs['ln'] = conv_out

        return conv_out

    def forward(self, obs, hist_obs, hist_act, hist_seg_len, detach=False):
        # step 1
        obs_after_encoder = self.forward_conv(obs, detach)

        # step 2
        hist_obs_after_encoder = self.forward_conv(hist_obs, detach)
        hist_obs_after_encoder_size = list(hist_obs_after_encoder.size())
        hist_act_size = list(hist_act.size())
        batch_size = int((hist_obs_after_encoder_size[0] / (hist_seg_len[0].cpu())).item())
        hist_len = int(hist_seg_len[0].item())
        hist_obs_after_encoder_seq = torch.zeros((batch_size, hist_len, hist_obs_after_encoder_size[-1])).to(self.device)
        hist_act_seq = torch.zeros((batch_size, hist_len, hist_act_size[-1])).to(self.device)

        if batch_size != 1:
            # update from the replay buffer
            hist_len_seq = torch.ones(batch_size).to(self.device)
            for i in range(batch_size):
                hist_obs_after_encoder_seq[i] = hist_obs_after_encoder[i*hist_len:(i+1)*hist_len]
                hist_act_seq[i] = hist_act[i * hist_len:(i + 1) * hist_len]
                hist_len_seq[i] = hist_seg_len[i * hist_len]
        else:
            # sample action from the obs
            hist_obs_after_encoder_seq = hist_obs_after_encoder.unsqueeze(0)
            hist_act_seq = hist_act.unsqueeze(0)
            hist_len_seq = hist_seg_len

        hist_out = self.lstm_encoder(hist_obs_after_encoder_seq, hist_act_seq, hist_len_seq)

        # Combination
        combination_obs = torch.cat([hist_out, obs_after_encoder], dim=-1)
        encoder_out = self.fc2(combination_obs)

        return encoder_out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

        # tie_weights(src=source.fc1, trg=self.fc1)
        # # tie_weights(src=source.fc2, trg=self.fc2)
        # tie_weights(src=source.ln, trg=self.ln)
        # tie_weights(src=source.lstm_encoder.fc1, trg=self.lstm_encoder.fc1)
        # tie_weights(src=source.lstm_encoder.fc2, trg=self.lstm_encoder.fc2)
        # self.lstm_encoder.lstm_module.weight_ih_l0 = source.lstm_encoder.lstm_module.weight_ih_l0
        # self.lstm_encoder.lstm_module.weight_hh_l0 = source.lstm_encoder.lstm_module.weight_hh_l0
        # self.lstm_encoder.lstm_module.bias_ih_l0 = source.lstm_encoder.lstm_module.bias_ih_l0
        # self.lstm_encoder.lstm_module.bias_hh_l0 = source.lstm_encoder.lstm_module.bias_hh_l0

        # for name, param in source.named_parameters():
        #     tie_weights(src=source.name, trg=self.name)
        #     # print("tie weights", name, param.size())

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc1, step)
        L.log_param('train_encoder/ln', self.ln, step)


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'PixelLstmEncoderCarla096': PixelLstmEncoderCarla096,        # added by 51
                       'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride, device, action_shape
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, stride, device, action_shape
    )
