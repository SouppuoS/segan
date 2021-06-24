import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as opt
from dataset import seganDataset
from torch.utils.data import DataLoader
from segan_module import d_module, g_module
from asteroid.losses import singlesrc_neg_sisdr, singlesrc_neg_snr

class segan():
    def __init__(self, device='cuda') -> None:
        self.Chncfg = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.device = device
        self.G = g_module(chnlCfg=[1] + self.Chncfg).to(self.device)
        self.D = d_module(chnlCfg=[2] + self.Chncfg).to(self.device)

    def train(self, tr_path, cv_path, epoch=200, batch_size=8):
        trndata   = seganDataset(tr_path, device=self.device)
        cvdata    = seganDataset(cv_path, device=self.device)
        trnDatLdr = DataLoader(trndata, batch_size=batch_size, shuffle=True, drop_last=True)
        vldDatLdr = DataLoader(cvdata,  batch_size=batch_size)

        epoch_max = epoch
        # clsLoss   = nn.BCELoss()
        clsLoss   = nn.MSELoss()
        l1Loss    = nn.L1Loss()
        # sgnLoss   = singlesrc_neg_sisdr
        sgnLoss   = singlesrc_neg_snr
        d_optim   = opt.Adam(self.D.parameters(), lr=1e-4)
        g_optim   = opt.Adam(self.G.parameters(), lr=1e-4)

        _lambda   = 200

        pos_tgt   = torch.tensor([1. for _ in range(batch_size)], device=self.device)
        neg_tgt   = torch.tensor([0. for _ in range(batch_size)], device=self.device)

        for ep in range(epoch_max):
            print(f'== epoch {ep} ==')
            self.G.train()
            losses_D = []
            losses_G = []
            for noisy, clean in tqdm(trnDatLdr):
                self.D.train()
                # step 1. pos sample, train D only
                ins_s1  = torch.cat((noisy[:, None, :], clean), dim=1)
                rst_s1  = self.D(ins_s1)

                # step 2. neg sample, train D only
                with torch.no_grad():
                    recon = self.G(noisy)
                ins_s2  = torch.cat((recon, clean), dim=1)
                rst_s2  = self.D(ins_s2)
                loss_D  = clsLoss(rst_s2[:, 0, 0], neg_tgt) + clsLoss(rst_s1[:, 0, 0], pos_tgt)

                d_optim.zero_grad()
                loss_D.backward()
                nn.utils.clip_grad.clip_grad_norm_(self.D.parameters(), 3)
                d_optim.step()

                # step 3. misclassfy, train G only
                self.D.eval()
                recon   = self.G(noisy)
                ins_s3  = torch.cat((recon, clean), dim=1)
                rst_s3  = self.D(ins_s3)
                loss_G  = clsLoss(rst_s3[:, 0, 0], pos_tgt) + _lambda * l1Loss(recon, clean)
                # loss_G  = clsLoss(rst_s3[:, 0, 0], pos_tgt) + _lambda * sgnLoss(recon[:, 0, :], clean[:, 0, :]).mean()

                g_optim.zero_grad()
                loss_G.backward()
                nn.utils.clip_grad.clip_grad_norm_(self.G.parameters(), 3)
                g_optim.step()

                losses_G.append(loss_G)
                losses_D.append(loss_D)

            self.G.eval()
            with torch.no_grad():
                losses = []
                for noisy, clean in tqdm(iter(vldDatLdr)):
                    recon   = self.G(noisy)
                    loss    = sgnLoss(recon[:, 0, :], clean[:, 0, :]).mean()
                    losses.append(loss)
            print('trn G loss = {:.4f} D loss = {:.4f} val loss = {:.4f}'.format(
                sum(losses_G) / len(losses_G), 
                sum(losses_D) / len(losses_D), 
                sum(losses) / len(losses)))

            if (ep + 1) % 5 == 0:
                torch.save(self.G, f'ckpt/model_G_{ep}.pt')
                torch.save(self.D, f'ckpt/model_D_{ep}.pt')

if __name__ == '__main__':
    path_data_json = {'tr': './data-enh-210526/1speakers/wav8k/min/tr',
                      'cv': './data-enh-210526/1speakers/wav8k/min/cv',
                      'tt': './data-enh-210526/1speakers/wav8k/min/tt',}
    model = segan(device='cuda')
    model.train(tr_path=path_data_json['tr'], cv_path=path_data_json['cv'], batch_size=24)