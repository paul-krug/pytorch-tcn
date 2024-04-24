import torch 
from torch.nn import functional as F


class FeatureLoss( torch.nn.Module ):
    def __init__(
            self,
            ):
        super().__init__()
        return

    def forward(
            self,
            fmap_r,
            fmap_g,
            ):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2 

class DiscriminatorLoss( torch.nn.Module ):
    def __init__(
            self,
            ):
        super().__init__()
        return
    
    def forward(
            self,
            disc_real_outputs,
            disc_generated_outputs,
            ):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        return loss, r_losses, g_losses

class GeneratorLoss( torch.nn.Module ):
    def __init__(
            self,
            ):
        super().__init__()
        return
    
    def generator_loss(
            self,
            disc_outputs,
            ):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

class KullbackLeiblerLoss( torch.nn.Module ):
    def __init__(
            self,
            ):
        super().__init__()
        return
    
    def forward(z_p, logs_q, m_p, logs_p, z_mask):
        """
        z_p, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()
    
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l