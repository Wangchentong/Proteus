import torch
import torch.nn as nn
from model.ipa_pytorch import Linear

class DistogramHead(nn.Module):

    def __init__(self, c_z, no_bins, asymmetry=False, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins
        self.asymmetry = asymmetry

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        if not self.asymmetry:
            logits = (logits + logits.transpose(-2, -3))/2
        return logits

class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()
        self.config = config

        self.dist_head = DistogramHead(**config.distogram_6d.dist)
        self.omega_head = DistogramHead(**config.distogram_6d.omega)
        self.theta_head = DistogramHead(**config.distogram_6d.theta,asymmetry = True)
        self.phi_head = DistogramHead(**config.distogram_6d.phi, asymmetry = True)

    def forward(self, s, z):
        aux_out = {}
        aux_out["dist6d_logits"] = self.dist_head(z)
        aux_out["omega6d_logits"] = self.omega_head(z)
        aux_out["theta6d_logits"] = self.theta_head(z)
        aux_out["phi6d_logits"] = self.phi_head(z)

        return aux_out