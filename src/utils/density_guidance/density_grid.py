import os.path
from copy import copy
from itertools import product
from struct import unpack as _unpack, pack as _pack
from sys import byteorder as _BYTEORDER
import logging
from scipy.ndimage import map_coordinates

import numpy as np

from .density_spacegroup import GetSpaceGroup
from .density_unitcell import UnitCell

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def to_tensor(data, dtype=torch.float32, device='cuda', requires_grad=True):
    if isinstance(data, (np.ndarray, list, tuple)):
        tensor = torch.tensor(data, dtype=dtype, device=device)
    elif isinstance(data, torch.Tensor):
        tensor = data.to(device=device, dtype=dtype).clone()
        if tensor.requires_grad != requires_grad:
            tensor = tensor.detach().requires_grad_(requires_grad)
    else:
        tensor = torch.tensor(data, dtype=dtype, device=device)
    return tensor

class GridParameters:
    def __init__(self, voxelspacing=(1, 1, 1), offset=(0, 0, 0), dtype=torch.float64, device='cuda'):
        self.dtype = dtype
        self.device = torch.device(device)
        
        if isinstance(voxelspacing, (int, float)):
            voxelspacing = [voxelspacing] * 3

        self.voxelspacing = to_tensor(voxelspacing, dtype=self.dtype, device=self.device, requires_grad=False)
        self.offset = to_tensor(offset, dtype=torch.int32, device=self.device, requires_grad=False)

    def copy(self):
        return GridParameters(self.voxelspacing.clone(), self.offset.clone(), dtype=self.dtype, device=self.device)

class Resolution:
    def __init__(self, high=None, low=None):
        self.high = high
        self.low = low

    def copy(self):
        return Resolution(self.high, self.low)


class _BaseVolume:
    def __init__(self, array, grid_parameters=None, origin=(0, 0, 0), dtype=torch.float64, device='cuda'):
        self.dtype = dtype
        self.device = torch.device(device)

        self.array = to_tensor(array, dtype=self.dtype, device=self.device, requires_grad=False)
        self.np_array = self.array.cpu().numpy()

        if grid_parameters is None:
            grid_parameters = GridParameters(dtype=self.dtype, device=self.device)

        self.grid_parameters = grid_parameters
        
        self.origin = to_tensor(origin, dtype=self.dtype, device=self.device, requires_grad=False)

    @property
    def shape(self):
        return self.array.shape

    @property
    def offset(self):
        return self.grid_parameters.offset

    @property
    def voxelspacing(self):
        return self.grid_parameters.voxelspacing

    def tofile(self, fid, fmt=None):
        if fmt is None:
            fmt = os.path.splitext(fid)[-1][1:]
        if fmt in ("ccp4", "map", "mrc"):
            to_mrc(fid, self)
        else:
            raise ValueError("Format is not supported.")

class XMap(_BaseVolume):

    """A periodic volume with a unit cell and space group."""

    def __init__(
        self,
        array,
        grid_parameters=None,
        unit_cell=None,
        resolution=None,
        hkl=None,
        origin=None,
        dtype=torch.float64,
        device="cuda:0"
    ):
        super().__init__(array, grid_parameters, dtype=dtype, device=device)

        self.unit_cell = unit_cell
        self.dtype = dtype
        self.hkl = to_tensor(hkl, dtype=torch.int32, device=self.device, requires_grad=False) if hkl is not None else None
        self.resolution = resolution
        self.cutoff_dict = {}
        if origin is None:
            self.origin = torch.zeros(3, dtype=self.dtype, device=self.device)
        else:
            self.origin = to_tensor(origin, dtype=self.dtype, device=self.device, requires_grad=False)

    @classmethod
    def fromfile(cls, fname, fmt=None, resolution=None, label="FWT,PHWT", dtype=torch.float64, device='cuda:0'):
        if fmt is None:
            fmt = os.path.splitext(fname)[1]
        if fmt in (".ccp4", ".mrc", ".map"):
            if resolution is None:
                raise ValueError(
                    f"{fname} is a CCP4/MRC/MAP file. Please provide a resolution (use the '-r'/'--resolution' flag)."
                )
            parser = parse_volume(fname, fmt=fmt)
            a, b, c = parser.abc
            alpha, beta, gamma = parser.angles
            spacegroup = parser.spacegroup
            if spacegroup == 0:
                raise RuntimeError(
                    f"File {fname} is 2D image or image stack. Please convert to a 3D map."
                )
            unit_cell = UnitCell(a, b, c, alpha, beta, gamma, spacegroup, dtype=dtype, device=device)
            offset = parser.offset
            array = parser.density
            voxelspacing = parser.voxelspacing
            grid_parameters = GridParameters(voxelspacing, offset, dtype=dtype, device=device)
            resolution = Resolution(high=resolution)
            origin = to_tensor(parser.origin, dtype=dtype, device=device, requires_grad=False)
            xmap = cls(
                array,
                grid_parameters,
                unit_cell=unit_cell,
                resolution=resolution,
                origin=origin,
                dtype=dtype,
                device=device
            )
        else:
            raise RuntimeError("File format not recognized.")
        return xmap

    @classmethod
    def zeros_like(cls, xmap):
        array = torch.zeros_like(xmap.array)
        try:
            uc = xmap.unit_cell.copy()
        except AttributeError:
            uc = None
        hkl = copy(xmap.hkl)
        return cls(
            array,
            grid_parameters=xmap.grid_parameters.copy(),
            unit_cell=uc,
            hkl=hkl,
            resolution=xmap.resolution.copy(),
            origin=xmap.origin.clone(),
            dtype=array.dtype,
            device=array.device
        )

    def asymmetric_unit_cell(self):
        raise NotImplementedError

    @property
    def unit_cell_shape(self):
        shape = torch.round(self.unit_cell.abc / self.grid_parameters.voxelspacing).int()
        return shape

    def extend_to_p1(self, grid, offset, symop, out):
        # Ensure all inputs are on the correct device and have the right dtype
        device = grid.device
        grid = grid.to(dtype=self.dtype, device=device)
        offset = offset.to(dtype=torch.int32, device=device)
        symop = symop.to(dtype=self.dtype, device=device).flatten()
        out = out.to(dtype=self.dtype, device=device)

        grid_shape = torch.tensor(grid.shape, dtype=torch.int32, device=device)
        out_shape = torch.tensor(out.shape, dtype=torch.int32, device=device)

        # Create coordinate grids
        z, y, x = torch.meshgrid(torch.arange(grid_shape[0], device=device),
                                torch.arange(grid_shape[1], device=device),
                                torch.arange(grid_shape[2], device=device),
                                indexing='ij')

        grid_slice, out_slice = grid_shape[2] * grid_shape[1], out_shape[2] * out_shape[1]

        # Add offset
        grid_z = z + offset[2]
        grid_y = y + offset[1]
        grid_x = x + offset[0]

        # Apply symmetry operation step by step
        out_z_z = (symop[11] + symop[10] * grid_z).long()
        out_y_z = (symop[7] + symop[6] * grid_z).long()
        out_x_z = (symop[3] + symop[2] * grid_z).long()
        
        out_z_zy = (out_z_z + symop[9] * grid_y).long()
        out_y_zy = (out_y_z + symop[5] * grid_y).long()
        out_x_zy = (out_x_z + symop[1] * grid_y).long()
        
        out_z = (out_z_zy + symop[8] * grid_x).long()
        out_y = (out_y_zy + symop[4] * grid_x).long()
        out_x = (out_x_zy + symop[0] * grid_x).long()
        
        grid_indices = z * grid_slice + y * grid_shape[2] + x
        
        # Apply modulo
        out_z = out_z % out_shape[0]
        out_y = out_y % out_shape[1]
        out_x = out_x % out_shape[2]
        
        # Compute flattened indices
        out_indices = (out_z * out_slice +
                    out_y * out_shape[2] +
                    out_x)
        
        out_cpu = out.clone().cpu()
        out_cpu.view(-1).scatter_(0, out_indices.view(-1).cpu(), grid.view(-1)[grid_indices.view(-1)].cpu())
        return out_cpu.to(device)

    def canonical_unit_cell(self):
        shape = torch.round(self.unit_cell.abc / self.grid_parameters.voxelspacing).to(torch.int64).flip(0)
        array = torch.zeros(shape.tolist(), device=self.device, dtype=self.dtype)
        grid_parameters = GridParameters(self.voxelspacing, device=self.device, dtype=self.dtype)
        out = XMap(
            array,
            grid_parameters=grid_parameters,
            unit_cell=self.unit_cell,
            hkl=self.hkl,
            resolution=self.resolution,
            device=self.device,
            dtype=self.dtype
        )
        offset = to_tensor(self.offset, dtype=torch.int32, device=self.device, requires_grad=False)
        for symop in self.unit_cell.space_group.symop_list:
            R_torch = to_tensor(symop.R, dtype=self.dtype, device=self.device)
            t_torch = to_tensor(symop.t, dtype=self.dtype, device=self.device)

            transform = torch.cat((R_torch, t_torch.reshape(3, -1)), dim=1)
            transform[:, -1] = transform[:, -1] * torch.tensor(out.shape[::-1], device=self.device).int()
            out.array = self.extend_to_p1(self.array, offset, transform, out.array)
        out.np_array = out.array.clone().cpu().numpy()
        return out

    def is_canonical_unit_cell(self):
        return torch.allclose(to_tensor(self.shape, device=self.device, dtype=torch.int), self.unit_cell_shape.flip(0)) and torch.allclose(
            self.offset, torch.zeros_like(self.offset, device=self.device)
        )

    def extract(self, orth_coor, padding=3.0):
        """Create a copy of the map around the atomic coordinates provided.

        Args:
            orth_coor (np.ndarray[(n_atoms, 3), dtype=np.float]):
                a collection of Cartesian atomic coordinates
            padding (float): amount of padding (in Angstrom) to add around the
                returned electron density map
        Returns:
            XMap: the new map object around the coordinates
        """
        if not self.is_canonical_unit_cell():
            raise RuntimeError("XMap should contain full unit cell.")

        # Convert atomic Cartesian coordinates to voxelgrid coordinates
        orth_coor = to_tensor(orth_coor, dtype=self.dtype, device=self.device)
        grid_coor = torch.matmul(orth_coor, self.unit_cell.orth_to_frac.T)
        grid_coor = grid_coor * self.unit_cell_shape
        grid_coor = grid_coor - self.offset

        # How many voxels are we padding by?
        grid_padding = padding / self.voxelspacing

        # What are the voxel-coords of the lower and upper extrema that we will extract?
        lb = grid_coor.min(dim=0)[0] - grid_padding
        ru = grid_coor.max(dim=0)[0] + grid_padding
        lb = torch.floor(lb).to(torch.int64)
        ru = torch.ceil(ru).to(torch.int64)
        shape = (ru - lb).flip(0)
        logger.debug(f"From old map size (voxels): {self.shape}")
        logger.debug(f"Extract between corners:    {lb.flip(0)}, {ru.flip(0)}")
        logger.debug(f"New map size (voxels):      {shape}")

        # Make new GridParameters, make sure to update offset
        grid_parameters = GridParameters(self.voxelspacing, self.offset + lb, dtype=self.dtype, device=self.device)
        offset = grid_parameters.offset

        # Get the ranges across all the axes
        ranges = [torch.arange(axis_len, device=self.device).clone().detach().cpu().numpy() for axis_len in shape]

        ixgrid = np.ix_(*ranges)
        ixgrid = tuple(
            (dimension_index + offset) % wrap_to
            for dimension_index, offset, wrap_to in zip(
                ixgrid, offset.clone().cpu().numpy()[::-1], self.unit_cell_shape.clone().cpu().numpy()[::-1]
            )
        )
        density_map = self.array[ixgrid]
        out_map = XMap(density_map, grid_parameters=grid_parameters, unit_cell=self.unit_cell,
                       resolution=self.resolution, hkl=self.hkl, origin=self.origin, device=self.device,
                       dtype=self.dtype)
        return out_map, ixgrid

    def coordinates_map(self, coords):
        """
        Function to take in coordinates and estimate density in the grid
        """
        return map_coordinates(self.np_array, coords, order=1)

    def transform_to_grid_coordinates(self, xyz):
        """
        Transform Cartesian coordinates to grid coordinates
        """
        batch_size, N, _ = xyz.shape
        uc = self.unit_cell
        orth_to_grid = uc.orth_to_frac * self.unit_cell_shape.reshape(3, 1)
        orth_to_grid = orth_to_grid.unsqueeze(0).expand(batch_size, -1, -1)

        if not torch.allclose(self.origin, torch.tensor(0.0, dtype=self.dtype, device=self.device)):
            origin = self.origin.unsqueeze(0).unsqueeze(1).expand(batch_size, N, -1)
            xyz = xyz - origin

        xyz_reshaped = xyz.transpose(1, 2)
        return torch.bmm(orth_to_grid, xyz_reshaped)

    def apply_offset_and_wrap(self, grid_coor):
        """
        Apply offset and wrap coordinates if in canonical unit cell
        """
        batch_size = grid_coor.shape[0]
        offset = self.offset.reshape(3, 1).unsqueeze(0).expand(batch_size, -1, -1)
        grid_coor = grid_coor - offset

        if self.is_canonical_unit_cell():
            unit_cell_shape = self.unit_cell_shape.reshape(3, 1).unsqueeze(0).expand(batch_size, -1, -1)
            grid_coor = grid_coor % unit_cell_shape

        return grid_coor

    def process_grid_coordinates(self, grid_coor):
        """
        Process grid coordinates for density interpolation
        """
        grid_coor_np = grid_coor.flip(1).detach().cpu().numpy()
        vect_map_coordinates = np.vectorize(self.coordinates_map, signature='(3,n)->(n)')
        density_values = vect_map_coordinates(grid_coor_np)
        return torch.tensor(density_values, device=self.device, dtype=self.dtype)

    def interpolate(self, xyz):
        """
        Given a batch of points xyz of shape [batch_size, N, 3], interpolate the density at those points
        Output shape [batch_size, N, 1]
        """
        grid_coor = self.transform_to_grid_coordinates(xyz)
        grid_coor = self.apply_offset_and_wrap(grid_coor)
        return self.process_grid_coordinates(grid_coor)

    def set_space_group(self, space_group):
        self.unit_cell.space_group = GetSpaceGroup(space_group)
        
    def get_voxels_cartisian_centeroids(self):
        """
            this function returns the centeroid of each voxel in the xmap
        """
        indexes = torch.stack(torch.meshgrid(*(torch.arange(0,self.shape[i], device=self.device) for i in range(len(self.shape))), indexing="ij"), dim=-1)
        indexes = indexes.flip(-1)
        indexes_shape = indexes.shape
        indexes = indexes.flatten(0,-2)
        
        grid_coords = indexes[None]
        uc = self.unit_cell
        # TODO: make sure this is the matrix we are supoosed to use
        lattice_to_cartesian = uc.frac_to_orth / uc.abc

        # Apply offset
        grid_coords = grid_coords + self.offset

        # Scale by voxel spacing
        grid_coords = grid_coords * self.voxelspacing

        # Convert to Cartesian coordinates
        cartesian_coordinates = torch.bmm(grid_coords, lattice_to_cartesian.T[None].expand(grid_coords.shape[0], -1, -1))

        # Apply origin shift if necessary
        if not torch.allclose(self.origin, torch.zeros_like(self.origin)):
            cartesian_coordinates = cartesian_coordinates + self.origin


        cartesian_coordinates = cartesian_coordinates.squeeze(0)
        return cartesian_coordinates.reshape(indexes_shape)

# Volume parsers
def parse_volume(fid, fmt=None):
    try:
        fname = fid.name
    except AttributeError:
        fname = fid

    if fmt is None:
        fmt = os.path.splitext(fname)[-1]
    if fmt == ".ccp4":
        p = CCP4Parser(fname)
    elif fmt in (".map", ".mrc"):
        p = MRCParser(fname)
    else:
        raise ValueError("Extension of file is not supported.")
    return p


class CCP4Parser:
    HEADER_SIZE = 1024
    HEADER_TYPE = (
        "i" * 10
        + "f" * 6
        + "i" * 3
        + "f" * 3
        + "i" * 3
        + "f" * 27
        + "c" * 8
        + "f" * 1
        + "i" * 1
        + "c" * 800
    )
    HEADER_FIELDS = (
        "nc nr ns mode ncstart nrstart nsstart nx ny nz xlength ylength "
        "zlength alpha beta gamma mapc mapr maps amin amax amean ispg "
        "nsymbt lskflg skwmat skwtrn extra xstart ystart zstart map "
        "machst rms nlabel label"
    ).split()
    HEADER_CHUNKS = [1] * 25 + [9, 3, 12] + [1] * 3 + [4, 4, 1, 1, 800]

    def __init__(self, fid):
        if isinstance(fid, str):
            fhandle = open(fid, "rb")
        elif isinstance(fid, file):
            fhandle = fid
        else:
            raise ValueError("Input should either be a file or filename.")

        self.fhandle = fhandle
        self.fname = fhandle.name

        # first determine the endiannes of the file
        self._get_endiannes()
        # get the header
        self._get_header()
        self.abc = tuple(self.header[key] for key in ("xlength", "ylength", "zlength"))
        self.angles = tuple(self.header[key] for key in ("alpha", "beta", "gamma"))
        self.shape = tuple(self.header[key] for key in ("nx", "ny", "nz"))
        self.voxelspacing = tuple(length / n for length, n in zip(self.abc, self.shape))
        self.spacegroup = int(self.header["ispg"])
        self.cell_shape = [self.header[key] for key in "nz ny nx".split()]
        self._get_offset()
        self._get_origin()
        # Get the symbol table and ultimately the density
        self._get_symbt()
        self._get_density()
        self.fhandle.close()

    def _get_endiannes(self):
        self.fhandle.seek(212)
        b = self.fhandle.read(1)

        m_stamp = hex(ord(b))
        if m_stamp == "0x44":
            endian = "<"
        elif m_stamp == "0x11":
            endian = ">"
        else:
            raise ValueError(
                "Endiannes is not properly set in file. Check the file format."
            )
        self._endian = endian
        self.fhandle.seek(0)

    def _get_header(self):
        header = _unpack(
            self._endian + self.HEADER_TYPE, self.fhandle.read(self.HEADER_SIZE)
        )
        self.header = {}
        index = 0
        for field, nchunks in zip(self.HEADER_FIELDS, self.HEADER_CHUNKS):
            end = index + nchunks
            if nchunks > 1:
                self.header[field] = header[index:end]
            else:
                self.header[field] = header[index]
            index = end
        self.header["label"] = "".join(x.decode("utf-8") for x in self.header["label"])

    def _get_offset(self):
        self.offset = [0] * 3
        self.offset[self.header["mapc"] - 1] = self.header["ncstart"]
        self.offset[self.header["mapr"] - 1] = self.header["nrstart"]
        self.offset[self.header["maps"] - 1] = self.header["nsstart"]

    def _get_origin(self):
        self.origin = (0, 0, 0)

    def _get_symbt(self):
        self.symbt = self.fhandle.read(self.header["nsymbt"])

    def _get_density(self):
        # Determine the dtype of the file based on the mode
        mode = self.header["mode"]
        if mode == 0:
            dtype = "i1"
        elif mode == 1:
            dtype = "i2"
        elif mode == 2:
            dtype = "f4"

        # Read the density
        storage_shape = tuple(self.header[key] for key in ("ns", "nr", "nc"))
        self.density = np.fromfile(self.fhandle, dtype=self._endian + dtype).reshape(
            storage_shape
        )

        # Reorder axis so that nx is fastest changing.
        maps, mapr, mapc = [self.header[key] for key in ("maps", "mapr", "mapc")]
        if maps == 3 and mapr == 2 and mapc == 1:
            pass
        elif maps == 3 and mapr == 1 and mapc == 2:
            self.density = np.swapaxes(self.density, 1, 2)
        elif maps == 2 and mapr == 1 and mapc == 3:
            self.density = np.swapaxes(self.density, 1, 2)
            self.density = np.swapaxes(self.density, 1, 0)
        elif maps == 1 and mapr == 2 and mapc == 3:
            self.density = np.swapaxes(self.density, 0, 2)
        else:
            msg = f"Density storage order ({maps} {mapr} {mapc}) not supported."
            raise NotImplementedError(msg)
        self.density = np.ascontiguousarray(self.density, dtype=np.float64)


class MRCParser(CCP4Parser):
    def _get_origin(self):
        origin_fields = "xstart ystart zstart".split()
        origin = [self.header[field] for field in origin_fields]
        self.origin = origin

def to_mrc(fid, volume, labels=[], fmt=None):
    if fmt is None:
        fmt = os.path.splitext(fid)[-1][1:]

    if fmt not in ("ccp4", "mrc", "map"):
        raise ValueError("Format is not recognized. Use ccp4, mrc, or map.")

    dtype = volume.array.dtype
    if dtype == torch.int8:
        mode = 0
    elif dtype in (torch.int16, torch.int32):
        mode = 1
    elif dtype in (torch.float32, torch.float64):
        mode = 2
    else:
        raise TypeError(f"Data type ({dtype}) is not supported.")

    if fmt == "ccp4":
        nxstart, nystart, nzstart = volume.offset
        origin = [0, 0, 0]
        uc = volume.unit_cell
        xl, yl, zl = uc.a.item(), uc.b.item(), uc.c.item()
        alpha, beta, gamma = uc.alpha.item(), uc.beta.item(), uc.gamma.item()
        ispg = uc.space_group.number
        ns, nr, nc = volume.unit_cell_shape.flip(0)
    elif fmt in ("mrc", "map"):
        nxstart, nystart, nzstart = [0, 0, 0]
        origin = volume.origin
        xl, yl, zl = [
            vs * n for vs, n in zip(volume.voxelspacing, reversed(volume.shape))
        ]
        alpha = beta = gamma = 90
        ispg = volume.unit_cell.space_group.number
        ns, nr, nc = volume.shape

    elif fmt in ("mrc", "map"):
        nxstart, nystart, nzstart = [0, 0, 0]
        origin = volume.origin.tolist()
        xl, yl, zl = [
            vs.item() * n for vs, n in zip(volume.voxelspacing, reversed(volume.shape))
        ]
        alpha = beta = gamma = 90
        ispg = 1
        ns, nr, nc = volume.shape
    voxelspacing = volume.voxelspacing
    nz, ny, nx = volume.shape
    mapc, mapr, maps = [1, 2, 3]
    nsymbt = 0
    lskflg = 0
    skwmat = [0.0] * 9
    skwtrn = [0.0] * 3
    fut_use = [0.0] * 12
    str_map = list("MAP ")
    str_map = "MAP "
    # TODO machst are similar for little and big endian
    if _BYTEORDER == "little":
        machst = list("\x44\x41\x00\x00")
    elif _BYTEORDER == "big":
        machst = list("\x44\x41\x00\x00")
    else:
        raise ValueError("Byteorder {:} is not recognized".format(_BYTEORDER))
    labels = [" "] * 800
    nlabels = 0
    min_density = volume.array.min().item()
    max_density = volume.array.max().item()
    mean_density = volume.array.mean().item()
    std_density = volume.array.std().item()

    with open(fid, "wb") as out:
        out.write(_pack("i", nx))
        out.write(_pack("i", ny))
        out.write(_pack("i", nz))
        out.write(_pack("i", mode))
        out.write(_pack("i", nxstart))
        out.write(_pack("i", nystart))
        out.write(_pack("i", nzstart))
        out.write(_pack("i", nc))
        out.write(_pack("i", nr))
        out.write(_pack("i", ns))
        out.write(_pack("f", xl))
        out.write(_pack("f", yl))
        out.write(_pack("f", zl))
        out.write(_pack("f", alpha))
        out.write(_pack("f", beta))
        out.write(_pack("f", gamma))
        out.write(_pack("i", mapc))
        out.write(_pack("i", mapr))
        out.write(_pack("i", maps))
        out.write(_pack("f", min_density))
        out.write(_pack("f", max_density))
        out.write(_pack("f", mean_density))
        out.write(_pack("i", ispg))
        out.write(_pack("i", nsymbt))
        out.write(_pack("i", lskflg))
        for f in skwmat:
            out.write(_pack("f", f))
        for f in skwtrn:
            out.write(_pack("f", f))
        for f in fut_use:
            out.write(_pack("f", f))
        for f in origin:
            out.write(_pack("f", f))
        for c in str_map:
            out.write(_pack("c", c.encode("ascii")))
        for c in machst:
            out.write(_pack("c", c.encode("ascii")))
        out.write(_pack("f", std_density))

        out.write(_pack("i", nlabels))
        for c in labels:
            out.write(_pack("c", c.encode("ascii")))
        # Write density
        modes = [torch.int8, torch.int16, torch.float32]
        volume.array.to(modes[mode]).clone().detach().cpu().numpy().tofile(out)
