import torch
import torch.nn.functional as F
from itertools import product

from . import density_spacegroup

class UnitCell:
    def __init__(self, a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0, space_group="P1", dtype=torch.float64, device='cuda'):
        self.device = torch.device(device)
        self.dtype = dtype
        self.a = torch.tensor(a, dtype=self.dtype, device=self.device)
        self.b = torch.tensor(b, dtype=self.dtype, device=self.device)
        self.c = torch.tensor(c, dtype=self.dtype, device=self.device)

        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)
        self.beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
        self.gamma = torch.tensor(gamma, dtype=self.dtype, device=self.device)

        self.set_space_group(space_group)

        self._sin_alpha = torch.sin(torch.deg2rad(self.alpha))
        self._sin_beta = torch.sin(torch.deg2rad(self.beta))
        self._sin_gamma = torch.sin(torch.deg2rad(self.gamma))

        self._cos_alpha = torch.cos(torch.deg2rad(self.alpha))
        self._cos_beta = torch.cos(torch.deg2rad(self.beta))
        self._cos_gamma = torch.cos(torch.deg2rad(self.gamma))

        self.orth_to_frac = self.calc_fractionalization_matrix()
        self.frac_to_orth = self.calc_orthogonalization_matrix()

    def __str__(self):
        return f"UnitCell(a={self.a.item():.6f}, b={self.b.item():.6f}, c={self.c.item():.6f}, alpha={self.alpha.item():.6f}, beta={self.beta.item():.6f}, gamma={self.gamma.item():.6f})"

    def copy(self):
        return UnitCell(
            self.a.item(), self.b.item(), self.c.item(),
            self.alpha.item(), self.beta.item(), self.gamma.item(),
            self.space_group.number, device=self.device, dtype=self.dtype
        )

    @property
    def abc(self):
        return torch.stack([self.a, self.b, self.c])

    def calc_v(self):
        return torch.sqrt(
            1
            - (self._cos_alpha * self._cos_alpha)
            - (self._cos_beta * self._cos_beta)
            - (self._cos_gamma * self._cos_gamma)
            + (2 * self._cos_alpha * self._cos_beta * self._cos_gamma)
        )

    def calc_volume(self):
        return self.a * self.b * self.c * self.calc_v()

    def calc_reciprocal_unit_cell(self):
        V = self.calc_volume()

        ra = (self.b * self.c * self._sin_alpha) / V
        rb = (self.a * self.c * self._sin_beta) / V
        rc = (self.a * self.b * self._sin_gamma) / V

        ralpha = torch.arccos(
            (self._cos_beta * self._cos_gamma - self._cos_alpha)
            / (self._sin_beta * self._sin_gamma)
        )
        rbeta = torch.arccos(
            (self._cos_alpha * self._cos_gamma - self._cos_beta)
            / (self._sin_alpha * self._sin_gamma)
        )
        rgamma = torch.arccos(
            (self._cos_alpha * self._cos_beta - self._cos_gamma)
            / (self._sin_alpha * self._sin_beta)
        )

        return UnitCell(ra, rb, rc, torch.rad2deg(ralpha), torch.rad2deg(rbeta), torch.rad2deg(rgamma), device=self.device)

    def calc_orthogonalization_matrix(self):
        v = self.calc_v()

        f11 = self.a
        f12 = self.b * self._cos_gamma
        f13 = self.c * self._cos_beta
        f22 = self.b * self._sin_gamma
        f23 = (self.c * (self._cos_alpha - self._cos_beta * self._cos_gamma)) / self._sin_gamma
        f33 = (self.c * v) / self._sin_gamma

        orth_to_frac = torch.tensor([
            [f11, f12, f13],
            [0.0, f22, f23],
            [0.0, 0.0, f33]
        ], dtype=self.dtype, device=self.device)

        return orth_to_frac

    def calc_fractionalization_matrix(self):
        v = self.calc_v()

        o11 = 1.0 / self.a
        o12 = -self._cos_gamma / (self.a * self._sin_gamma)
        o13 = (self._cos_gamma * self._cos_alpha - self._cos_beta) / (self.a * v * self._sin_gamma)
        o22 = 1.0 / (self.b * self._sin_gamma)
        o23 = (self._cos_gamma * self._cos_beta - self._cos_alpha) / (self.b * v * self._sin_gamma)
        o33 = self._sin_gamma / (self.c * v)

        frac_to_orth = torch.tensor([
            [o11, o12, o13],
            [0.0, o22, o23],
            [0.0, 0.0, o33]
        ], dtype=self.dtype, device=self.device)

        return frac_to_orth

    def calc_orth_to_frac(self, v):
        return torch.matmul(self.orth_to_frac, v)

    def calc_frac_to_orth(self, v):
        return torch.matmul(self.frac_to_orth, v)

    def calc_orth_symop(self, symop):
        symop_R = torch.tensor(symop.R, dtype=self.dtype, device=self.device)
        symop_t = torch.tensor(symop.t, dtype=self.dtype, device=self.device)

        RF = torch.matmul(symop_R, self.orth_to_frac)
        ORF = torch.matmul(self.frac_to_orth, RF)
        Ot = torch.matmul(self.frac_to_orth, symop_t)
        return density_spacegroup.SymOp(ORF, Ot)

    def calc_orth_symop2(self, symop):
        symop_R = torch.tensor(symop.R, dtype=self.dtype, device=self.device)
        symop_t = torch.tensor(symop.t, dtype=self.dtype, device=self.device)

        RF = torch.matmul(symop_R, self.orth_to_frac)
        ORF = torch.matmul(self.frac_to_orth, RF)
        Rt = torch.matmul(symop_R, symop_t)
        ORt = torch.matmul(self.frac_to_orth, Rt)
        return density_spacegroup.SymOp(ORF, ORt)

    def calc_cell(self, xyz):
        cx = torch.where(xyz[0] < 0.0, torch.floor(xyz[0] - 1.0), torch.floor(xyz[0] + 1.0))
        cy = torch.where(xyz[1] < 0.0, torch.floor(xyz[1] - 1.0), torch.floor(xyz[1] + 1.0))
        cz = torch.where(xyz[2] < 0.0, torch.floor(xyz[2] - 1.0), torch.floor(xyz[2] + 1.0))
        return (cx.int(), cy.int(), cz.int())

    def set_space_group(self, space_group):
        self.space_group = density_spacegroup.GetSpaceGroup(space_group)
