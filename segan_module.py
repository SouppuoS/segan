import torch
import torch.nn as nn

class wavnetlike(nn.Module):
    def __init__(self, stride=2, kernal_size=31, channel_size=[], rev=False):
        super(wavnetlike, self).__init__()

        self.num_layer = len(channel_size) - 1
        self.cnn       = nn.ModuleList([])
        self.skipTns   = nn.ModuleList([])
        for i in range(self.num_layer):
            if rev:
                self.cnn.append(nn.ConvTranspose1d(in_channels=channel_size[i + 1],
                                                   out_channels=channel_size[i], 
                                                   kernel_size=kernal_size, 
                                                   stride=stride, 
                                                   padding=kernal_size // 2,
                                                   output_padding=1))
                self.skipTns.append(nn.Conv1d(in_channels=channel_size[i + 1] // 2,
                                              out_channels=channel_size[i + 1], 
                                              kernel_size=1, 
                                              stride=1))
            else:
                self.cnn.append(nn.Conv1d(in_channels=channel_size[i],
                                          out_channels=channel_size[i + 1], 
                                          kernel_size=kernal_size, 
                                          stride=stride, 
                                          padding=kernal_size // 2))

    def forward(self, inputs):
        raise NotImplementedError

class g_module(nn.Module):
    def __init__(self, chnlCfg):
        super(g_module, self).__init__()
        self.cfg = chnlCfg
        self.enc = wavnetlike(channel_size=chnlCfg)
        self.dec = wavnetlike(channel_size=[1] + [2 * v for v in chnlCfg[1:]], rev=True)
        self.act = nn.PReLU()

    def forward(self, inputs):
        # inputs, [B, T]
        output = inputs[:, None, :]
        outs   = []
        for cnn in self.enc.cnn:
            output = self.act(cnn(output))
            outs.append(output)

        # "sample the noise samples z from our prior 8×1024-dimensional normal distribu-tion N(0, I). "
        # "skip connections and the addition of the latent vector make the number of feature maps in every layer to be doubled"
        z      = torch.randn(outs[-1].shape, device=outs[-1].device)
        output = torch.cat((outs[-1], z), dim=1)

        output = self.act(self.dec.cnn[-1](output))

        for skip, cnn, trn in zip(outs[-2::-1], self.dec.cnn[-2::-1], self.dec.skipTns[-2::-1]):
            output = self.act(cnn(output + trn(skip)))

        return output

class d_module(nn.Module):
    def __init__(self, chnlCfg):
        super(d_module, self).__init__()
        self.cfg = chnlCfg
        self.enc = wavnetlike(channel_size=chnlCfg)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.vbn = nn.ModuleList([nn.BatchNorm1d(ch) for ch in chnlCfg[1:]])
        self.cls = nn.Sequential(
            nn.Conv1d(1024, 1, 1),
            nn.LeakyReLU(0.3),
            nn.Linear(8, 1),
        )
        self.prd = nn.Sigmoid()

    def forward(self, inputs):
        # inputs, [B, 2, T] -> [B, 1024, 8]
        output = inputs
        for bn, cnn in zip(self.vbn, self.enc.cnn):
            output = self.act(bn(cnn(output)))
        d = self.prd(self.cls(output))
        return d