"""
DCT operations by Jeongsoo Park

Collection of functional augmentations for DCT coefficients of format: (C, H, W, KH, KW)
    C: Channels, H,W: Height, Width pos of block, KH, KW: Kernel size (8,8 for 8x8 DCT coeffs)
"""

import torch
import einops
import math
from torchvision.transforms import functional as F
import utils.dct_torch_utils as dtu
import scipy.signal

def dct_prep(coeff: torch.Tensor, mode='start'):
    """
    preps dct coeff for augmentation

    if tuple -> list
    if element -> list (one item)

    returns in original data structure

    Args:
        coeff: DCT coefficient
        mode: 'start' or 'end'
    """
    if mode=='start':
        if type(coeff) == tuple: # if tuple, convert to list for in-place op
            coeff = list(coeff)
        else: # if not, create single item list
            coeff = [coeff]
        for i in range(len(coeff)):
            coeff[i] = coeff[i].clamp(min=-2**10, max=2**10-8) # clamp to 8-bit max DCT coeff range (they can possibly go over those after resize/flip/rotate/etc augmentation)'NoneType' object is not subscriptable
        return coeff
    elif mode=='end':
        if len(coeff) > 1: # if tuple->list, convert it back to tuple
            return tuple(coeff)
        else: # if single item, return single item without list wrapper
            return coeff[0]

def resize_dct_rudimentary(coeff: torch.Tensor, size):
    """
    Functional version of resize_dct
    WARNING: This is *NOT* a correct implementation -- This performs block-wise interpolation. The larger the KH, KW, the worse the resize quality.
    
    Args:
        coeff: dct coefficient with five dimensions (c, h, w, kh, kw)
        size: desired size of the dct coefficients after transformation

    Returns:
        Resized dct coefficient
    """
    c, _, _, kh, kw = coeff.shape
    coeff = einops.rearrange(coeff, "c h w kh kw -> (c kh kw) h w")
    coeff = F.resize(coeff, size=size) # (c kh kw) size, size
    coeff = einops.rearrange(coeff, "(c kh kw) h w -> c h w kh kw", c=c, kh=kh, kw=kw)
    return coeff

## FFTshift-like ops

def blockshift(coeff, dim=(-2, -1)):
    """
    shifts input 'coeff' so that the origin is moved to the center
    ex) |A B|     |D C|
        |C D| ->  |B A|
    """
    shifted_block = torch.zeros_like(coeff)
    H = coeff.shape[dim[0]]
    W = coeff.shape[dim[1]]
    topH = H//2
    leftW = W//2

    shifted_block = torch.roll(coeff, topH, dim[0])
    shifted_block = torch.roll(shifted_block, leftW, dim[1])

    return shifted_block

def iblockshift(coeff, dim=(-2, -1)):
    """
    Inverse-shifts input 'coeff' so that the origin is moved to the center
    ex) |D C|     |A B|
        |B A| ->  |C D|
    """
    shifted_block = torch.zeros_like(coeff)
    H = coeff.shape[dim[0]]
    W = coeff.shape[dim[1]]
    topH = H//2
    leftW = W//2
    bottomH = H-topH
    rightW = W-leftW

    shifted_block = torch.roll(coeff, bottomH, dim[0])
    shifted_block = torch.roll(shifted_block, rightW, dim[1])

    return shifted_block

## 90-degrees rotation for DCT
def rotate_dct_90deg(coeff, rotate=0):
    """
    Rotates input dct coefficinet by 90 degrees

    Inputs:
        coeff: dct coefficient (C, H, W, KH, KW)
        rotate: rotate 90 degrees (counter-clockwise) by this amount
    """
    assert len(coeff.shape) == 5, "This function is currently only implemented for 5-D standard DCT coefficient (C, H, W, KH, KW)"
    
    C, H, W, KH, KW = coeff.shape
    coeff = coeff.clone()
    rotate_sign = rotate/abs(rotate) if rotate != 0 else 1 # sign for rotation

    if abs(rotate) % 4 == 0: # 360 degrees rotation.
        return coeff 
    elif rotate_sign * (abs(rotate) % 4) == 3 or rotate_sign * (abs(rotate) % 4) == -1:
        # 90 degrees rotation, clockwise
        coeff = torch.rot90(coeff, k=-1, dims=(1, 2))
        coeff = coeff.transpose(-2, -1)
        coeff = flip_dct(coeff, direction='horizontal', fixed_pos=True)
    elif abs(rotate) % 4 == 2:
        # 180 degrees rotation
        coeff = flip_dct(coeff, direction='vertical')
        coeff = flip_dct(coeff, direction='horizontal')
    elif rotate_sign * (abs(rotate) % 4) == 1 or rotate_sign * (abs(rotate) % 4) == -3:
        # 90 degrees rotation, counter-clockwise
        coeff = torch.rot90(coeff, k=1, dims=(1, 2))
        coeff = coeff.transpose(-2, -1)
        coeff = flip_dct(coeff, direction='vertical', fixed_pos=True)

    return coeff


## DFT -> DCT transform

def generate_fourier_basis(length=8, scale=True, dtype=torch.complex64, device='cpu'):
    """
    Generates basis matrix for DFT coefficients
    """
    basis_h = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(dim=1) # L, 1 # height is t
    basis_w = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(dim=0) # 1, L # width is k
    basis = basis_h.mm(basis_w)
    basis = basis * (-1j*2*torch.pi/length)
    basis = basis.exp()
    if scale:
        basis = basis / (length**0.5)
    return basis

## Resize related ops start

def generate_basis_matrix(length=8, scale_add=0, scale_mult=1, scale=True, dtype=torch.float32, device='cpu'):
    """
    Generates basis matrix b (b1) of DCT coefficients
    scale: make it orthonormal (you want this! False only for testing)
    """
    basis_h = torch.arange(length, device=device, dtype=dtype).unsqueeze(dim=1) # L, 1
    basis_w = torch.arange(length, device=device, dtype=dtype).unsqueeze(dim=0) # 1, L
    basis_w += 0.5
    basis = basis_h.mm(basis_w)
    basis = basis * torch.pi/length
    
    if scale_add != 0:
        basis += scale_add # custom add scaling to params
    if scale_mult != 1:
        basis *= scale_mult # custom mult scaling to params
    basis = basis.cos()
    if scale:
        basis[0] *= 1/(2**0.5)
        basis *= (2 / length)**0.5
    return basis

def expand_basis_matrix_blockwise(basis, mult=8):
    """
    Arranges basis matrix into identity blocks
    i.e. |B 0|
         |0 B| for mult=2, B=basis
    """
    basis_expanded = basis.unsqueeze(0).repeat(mult,1,1)
    return torch.block_diag(*basis_expanded) # unpacks basis_expanded

def generate_conversion_matrix(
    length_small=2, mult=2, scale=True, dtype=torch.float32, device='cpu'
    ):
    """
    Generates conversion matrix
    This matrix is used to project smaller DCT blocks into large DCT block
    i.e. |B 0|      |B2  | (B=NxN, B2=2Nx2N in this example)
         |0 B|  ->  |    |

    Inputs:
        length_small: dct coefficient size of smaller DCT base
        mult: length of larger dct coefficient = length_small * mult
        scale: If True, the generated basis are orthonormal, allowing for easier calculation of inverse basis matrix
    """
    if mult==1: # if mult==1 then there is no need to generate basis matrix: return I
        return torch.eye(length_small, dtype=dtype, device=device)

    large_basis = generate_basis_matrix(length_small * mult, scale=scale, dtype=dtype, device=device)
    small_basis = generate_basis_matrix(length_small, scale=scale, dtype=dtype, device=device)
    
    small_blocks = expand_basis_matrix_blockwise(small_basis, mult)
    if scale != False:
        small_blocks_inv = small_blocks.T
    else:
        small_blocks_inv = torch.linalg.inv(small_blocks)

    conversion_matrix = large_basis.mm(small_blocks_inv)

    return conversion_matrix

def generate_conversion_matrix_dft(
    length_small=2, mult=2, scale=True, dtype=torch.float32, device='cpu'
    ):
    """
    Generates conversion matrix
    This matrix is used to project smaller DCT blocks into large *DFT* block
    i.e. |B 0|      |B2  | (B=NxN, B2=2Nx2N in this example)
         |0 B|  ->  |    |

    Inputs:
        length_small: dct coefficient size of smaller DCT base
        mult: length of larger dct coefficient = length_small * mult
        scale: If True, the generated bases are orthonormal, allowing for easier calculation of inverse basis matrix
    """
    large_basis = generate_fourier_basis(length_small * mult, scale=scale, dtype=dtype, device=device)
    small_basis = generate_basis_matrix(length_small, scale=scale, dtype=dtype, device=device)
    
    small_blocks = expand_basis_matrix_blockwise(small_basis, mult)
    if scale != False:
        small_blocks_inv = small_blocks.T
    else:
        small_blocks_inv = torch.linalg.inv(small_blocks)

    conversion_matrix = large_basis.mm(small_blocks_inv.to(torch.complex64))

    return conversion_matrix

def combine_blocks(coeff: torch.Tensor, conv_L=None, conv_M=None):
    """
    Combines DCT blocks to a one single large one (For rotation test)
    Input:
        coeff: DCT coefficient to combine (C, H, W, KH, KW)
        conv_L, conv_M: use this conversion matrix if given
    
    Output:
        combined DCT coefficient (C, H*KH, W*KW)
        conv_L, conv_M
    
    Warning: This function automatically casts coeff to float32
    """
    _, H, W, KH, KW = coeff.shape
    if conv_L == None:
        conversion_matrix_L = generate_conversion_matrix(length_small=KH, mult=H, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_L = conv_L
    if conv_M == None:        
        if H==W and KH==KW:
            conversion_matrix_M = conversion_matrix_L
        else:
            conversion_matrix_M = generate_conversion_matrix(length_small=KW, mult=W, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_M = conv_M

    coeff_comb = einops.rearrange(coeff.to(torch.float32), 'c h w kh kw -> c (h kh) (w kw)')
    coeff_composed = torch.einsum('h o, c o w -> c h w', conversion_matrix_L, coeff_comb)
    coeff_composed = torch.einsum('c h o, o w -> c h w', coeff_composed, conversion_matrix_M.T)

    return coeff_composed, conversion_matrix_L, conversion_matrix_M

def decompose_block(coeff: torch.Tensor, H, W, KH, KW, conv_L=None, conv_M=None):
    """
    Decomposes (H KH) x (W KW) block to (H, W, KH, KW)

    Input:
        coeff: DCT coefficient of (C, (H KH), (W KW))
        H, W, KH, KW: Dimension of decomposed DCT coeff
        conv_L, conv_M: use this conversion matrix if given
    
    Output:
        Decomposed DCT coefficient of (C, H, W, KH, KW)
        conv_L, conv_M
    
    Warning: This function automatically casts coeff to float32
    """    
    if conv_L == None:
        conversion_matrix_L = generate_conversion_matrix(length_small=KH, mult=H, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_L = conv_L
    if conv_M == None:        
        if H==W and KH==KW:
            conversion_matrix_M = conversion_matrix_L
        else:
            conversion_matrix_M = generate_conversion_matrix(length_small=KW, mult=W, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_M = conv_M

    coeff_decomp = torch.einsum('h o, c o w -> c h w', conversion_matrix_L.T, coeff.to(torch.float32))
    coeff_decomp = torch.einsum('c h o, o w -> c h w', coeff_decomp, conversion_matrix_M)

    coeff_decomp = einops.rearrange(coeff_decomp, 'c (h kh) (w kw) -> c h w kh kw', h=H, w=W)

    return coeff_decomp, conv_L, conv_M

def combine_blocks_dft(coeff: torch.Tensor, conv_L=None, conv_M=None):
    """
    Combines DCT blocks to a one single large DFT block (For rotation test)
    Input:
        coeff: DCT coefficient to combine (C, H, W, KH, KW)
        conv_L, conv_M: use this conversion matrix if given
    
    Output:
        combined DCT coefficient (C, H*KH, W*KW)
    
    Warning: This function automatically casts coeff to complex64
    """
    _, H, W, KH, KW = coeff.shape
    if conv_L == None:
        conversion_matrix_L = generate_conversion_matrix_dft(length_small=KH, mult=H, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_L = conv_L
    if conv_M == None:        
        if H==W and KH==KW:
            conversion_matrix_M = conversion_matrix_L
        else:
            conversion_matrix_M = generate_conversion_matrix_dft(length_small=KW, mult=W, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_M = conv_M
        
    coeff_comb = einops.rearrange(coeff.to(torch.complex64), 'c h w kh kw -> c (h kh) (w kw)')
    coeff_composed = torch.einsum('h o, c o w -> c h w', conversion_matrix_L, coeff_comb) * ((KH*H)**0.5)
    coeff_composed = torch.einsum('c h o, o w -> c h w', coeff_composed, torch.conj(conversion_matrix_M.T)) / ((KW*W)**0.5)

    return coeff_composed, conversion_matrix_L, conversion_matrix_M

def decompose_block_dft(coeff: torch.Tensor, H, W, KH, KW, conv_L=None, conv_M=None):
    """
    Decomposes DFT (H KH) x (W KW) block to DCT (H, W, KH, KW)

    Input:
        coeff: DCT coefficient of (C, (H KH), (W KW))
        H, W, KH, KW: Dimension of decomposed DCT coeff
        conv_L, conv_M: use this conversion matrix if given
    
    Output:
        Decomposed DCT coefficient of (C, H, W, KH, KW)
    
    Warning: This function automatically casts coeff to complex64
    """    
    if conv_L == None:
        conversion_matrix_L = generate_conversion_matrix_dft(length_small=KH, mult=H, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_L = conv_L
    if conv_M == None:        
        if H==W and KH==KW:
            conversion_matrix_M = conversion_matrix_L
        else:
            conversion_matrix_M = generate_conversion_matrix_dft(length_small=KW, mult=W, scale=True, dtype=torch.float32, device=coeff.device)
    else:
        conversion_matrix_M = conv_M

    coeff_decomp = torch.einsum('h o, c o w -> c h w', torch.conj(conversion_matrix_L.T), coeff.to(torch.complex64)) / ((KH*H)**0.5)
    coeff_decomp = torch.einsum('c h o, o w -> c h w', coeff_decomp, conversion_matrix_M) * ((KW*W)**0.5)

    coeff_decomp = einops.rearrange(coeff_decomp, 'c (h kh) (w kw) -> c h w kh kw', h=H, w=W)

    return coeff_decomp.real, conversion_matrix_L, conversion_matrix_M

def rotate_block(coeff: torch.Tensor, degrees=45, conv_L=None, conv_M=None, dtype='keep', window=False, pad=False):
    """
    Rotates DCT coefficient by degrees counter-clockwise

    Algorithm: 
        1. Combine blockwise DCT coeff (C, H, W, KH, KW) -> C, (H KH), (W, KW)
        2. Rotate combined DCT coeff using dct2dft rotation method
        3. Decompose combined DCT coeff (C, (H KH), (W KW)) to (C, H, W, KH, KW)

    Input:
        coeff: DCT coefficient (C, H, W, KH, KW)
        degrees: Degrees to rotate
        conv_L: conversion matrix for height (L, H -> (L*H))
        conv_M: conversion matrix for width (W, M -> (W*M))
        dtype: if 'keep', keep coeff dtype. else, cast to dtype
        window: Use tukey window per block if true
        pad: Size of padding (if false, no padding). Ex: 3 = 1 pad, 1 img, 1 pad on each sides.
    """
    C, H, W, KH, KW = coeff.shape
    coeff_dtype = coeff.dtype

    if pad: # padding to 2*(H,W) yields better result but is 4x slower than no padding. sqrt(2)*(H,W) is about twice slower than no padding. 3 * (H, W) gives the best result with 9x slower result
        assert pad >= 1, "Padding should be larger than 1"
        coeff_padded = torch.zeros((C, int(H*pad//1), int(W*pad//1), KH, KW), dtype=coeff.dtype, device=coeff.device)
        Hmargin = (int(H*pad//1)-H)//2
        Wmargin = (int(W*pad//1)-W)//2
        coeff_padded[:, Hmargin:Hmargin+H, Wmargin:Wmargin+W, :, :] = coeff
    else:
        coeff_padded=coeff

    _, Hp, Wp, _, _= coeff_padded.shape
    
    if window: # tukey alpha=0: rectangular window, alpha=1 = Hann window
        window_H = torch.tensor(scipy.signal.tukey(Hp, alpha=0.4), dtype=torch.float32).unsqueeze(dim=1)
        window_W = torch.tensor(scipy.signal.tukey(Wp, alpha=0.4), dtype=torch.float32).unsqueeze(dim=0)
        window_mat = window_H.mm(window_W)
        coeff_padded = coeff_padded.to(torch.float32) * window_mat.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) # element-wise multiplication

    # determine how many 90 degrees rotations are needed (so that lossy rotations are kept within -45 ~ 45 degrees)
    degrees_sign = degrees / abs(degrees) if degrees != 0 else 1
    deg_remainder = degrees_sign * ((abs(degrees)) % 360) # remaining degrees after removing full rotations
    deg_in_positive = (deg_remainder + 360) # make degrees positive
    deg_shifted = (deg_in_positive + 45) % 360 # 0-90 (-45~45 before +45 shift): no rotation, 90-180: rotate 90 deg once, 180-270: twice, 270:360: thrice
    rot90s = deg_shifted // 90 # number of 90 degrees rotations
    deg_leftover = -((rot90s * 90) - (deg_shifted - 45)) # how much degrees to rotate from transposed position
    
    coeff_padded = rotate_dct_90deg(coeff_padded, rotate=rot90s) # do transpose-based rotation (lossless)
    degrees = deg_leftover

    coeff_padded = blockshift(coeff_padded, dim=(1, 2))
    
    coeff_composed, conv_L, conv_M = combine_blocks_dft(coeff_padded, conv_L, conv_M)
    coeff_dft_rotated = dtu.rotate_dft_2d_spatial(coeff_composed, degrees=degrees, real=False, use_blockshift=False)
    coeff_decomposed, _, _ = decompose_block_dft(coeff_dft_rotated, Hp, Wp, KH, KW, conv_L, conv_M)
    
    coeff_decomposed = iblockshift(coeff_decomposed, dim=(1, 2))

    if pad: #unpad
        coeff_decomposed = coeff_decomposed[:, Hmargin:Hmargin+H, Wmargin:Wmargin+W, :, :]
    
    if dtype=='keep':
        if coeff_dtype in [torch.int16, torch.int8, torch.uint8]:
            coeff_decomposed = torch.round(coeff_decomposed)
        return coeff_decomposed.to(coeff_dtype), conv_L, conv_M
    else:
        if dtype in [torch.int16, torch.int8, torch.uint8]:
            coeff_decomposed = torch.round(coeff_decomposed)
        return coeff_decomposed.to(dtype), conv_L, conv_M

def upsample_dct(coeff: torch.Tensor, L: int=1, M: int=1, dtype=torch.float32, conv_L=None, conv_M=None):
    """
    Upsamples DCT coefficient by L X M (Height by L, Width by M)
    Input:
        coeff (Tensor): DCT coefficient to upsample (C, H, W, KH, KW)
        L: Factor to upsample height by
        M: Factor to upsample width by
        dtype: Dtype during computation (float32 or bfloat16)
    
    Output:
        upsampled coefficient

    Warning: This function automatically casts coeff to dtype
    """
    assert L >= 1, "L should be a positive integer"
    assert M >= 1, "M should be a positive integer"
    assert len(coeff.shape) == 5, "DCT coefficients should be 5-dimensional: (C, H, W, KH, KW)"
    
    if L==M and M==1: # if input==output size
        return coeff.to(dtype), conv_L, conv_M

    _, H, W, KH, KW = coeff.shape
    upsamp_shape = list(coeff.shape) # shape after upsampling
    upsamp_shape[-2] = L*KH
    upsamp_shape[-1] = M*KW
    if conv_L==None:
        conversion_matrix_L = generate_conversion_matrix(length_small=KH, mult=L, scale=True, dtype=dtype, device=coeff.device)
    else:
        conversion_matrix_L = conv_L
    if conv_M==None:
        if L==M and KH==KW:
            conversion_matrix_M = conversion_matrix_L
        else:
            conversion_matrix_M = generate_conversion_matrix(length_small=KW, mult=M, scale=True, dtype=dtype, device=coeff.device)    
    else:
        conversion_matrix_M = conv_M
    
    # upsample KHxKW DCT blocks to (L KH)x(M KW) block by padding & scaling
    coeff_upsamp_approx = torch.zeros(tuple(upsamp_shape), dtype=dtype, device=coeff.device)
    coeff_upsamp_approx[..., 0:KH, 0:KW] = coeff.to(dtype) * (L*M)**0.5

    # Decompose (L KH)x(M KW) block into LxM NxN blocks
    coeff_upsamp_decomposed = torch.einsum("l o, c h w o m -> c h w l m", conversion_matrix_L.T, coeff_upsamp_approx)
    coeff_upsamp_decomposed = torch.einsum("c h w l o, o m -> c h w l m", coeff_upsamp_decomposed, conversion_matrix_M)
    
    # einops to rearrange them. Since dimension isn't fixed, this is actually a bit difficult. Prolly a fixed # of dim is needed. Could do using squeeze/unsqueeze tactic of torchvision
    return einops.rearrange(coeff_upsamp_decomposed, "c h w (l kh) (m kw) -> c (h l) (w m) kh kw", l=L, m=M), conversion_matrix_L, conversion_matrix_M

def downsample_dct(coeff: torch.Tensor, L: int=1, M: int=1, dtype=torch.float32, conv_L=None, conv_M=None):
    """
    Downsamples DCT coefficient by L X M (Height by L, Width by M)
    Input:
        coeff (Tensor): DCT coefficient to downsample (C, H, W, KH, KW)
        L: Factor to downsample height by
        M: Factor to downsample width by
        dtype: Dtype during computation (default= torch.float32, choose between float32 and torch.bfloat16)
    
    Output:
        downsample coefficient
    
    Warning: This function automatically casts coeff to float32
    """
    assert L >= 1, "L should be a positive integer"
    assert M >= 1, "M should be a positive integer"
    assert len(coeff.shape) == 5, "DCT coefficients should be 5-dimensional: (C, H, W, KH, KW)"
    
    if L==M and M==1: # if input==output size
        return coeff.to(dtype), conv_L, conv_M

    _, H, W, KH, KW = coeff.shape
    if conv_L==None:
        conversion_matrix_L = generate_conversion_matrix(length_small=KH, mult=L, scale=True, dtype=dtype, device=coeff.device)
    else:
        conversion_matrix_L = conv_L
    if conv_M==None:
        if L==M and KH==KW:
            conversion_matrix_M = conversion_matrix_L
        else:
            conversion_matrix_M = generate_conversion_matrix(length_small=KW, mult=M, scale=True, dtype=dtype, device=coeff.device)    
    else:
        conversion_matrix_M = conv_M

    # combine LxM KHxKW blocks to (L KH)x(M KW) blocks (block-wise)
    coeff_comb = einops.rearrange(coeff.to(dtype), 'c (h l) (w m) kh kw -> c h w (l kh) (m kw)', l=L, m=M)

    # combine blocks into one large (L KH)x(M KW) DCT coefficient. Notice the 'transpose' difference vs upsampling one. This is due to conversion matrix being orthonormal. i.e. CC' = I
    coeff_composed = torch.einsum("l o, c h w o m -> c h w l m", conversion_matrix_L, coeff_comb)
    coeff_composed = torch.einsum("c h w l o, o m -> c h w l m", coeff_composed, conversion_matrix_M.T)

    # approximate NxN blocks from LNxMN DCT coefficient
    coeff_downsampled = coeff_composed[..., 0:KH, 0:KW] / (L*M)**0.5
    return coeff_downsampled, conversion_matrix_L, conversion_matrix_M

def resize_dct(coeff: torch.Tensor, size, dtype=torch.float32, dtype_out='keep', conv_mxs :dict={}):
    """
    Resizes DCT coefficient to (size x size) block
    Can be very expensive depending on the resize ratio (ex. 100x100 -> 50x50: downsize by 2: cheaper. 100x100 -> 101x101: upsample by 101, downsample by 100: very expensive)

    Input:
        coeff: DCT coefficient of (C, H, W, KH, KW)
        size: Size to scale DCT coefficient into (Num blocks. H, W will be scaled to this value)
        dtype: dtype used during computation (default=torch.float32)
        dtype_out: 'keep' or desired dtype. If 'keep', it keeps input's dtype. Otherwise, it casts to dtype
        conv_Ls, conv_Ms: list of conversion matrices (idx 0: upsample, idx 1: downsample)
    """
    C, H, W, KH, KW = coeff.shape
    coeff_dtype = coeff.dtype
    
    # We can only scale the coefficients using integer multiples
    h_gcd = math.gcd(H, size)
    w_gcd = math.gcd(W, size)
    
    h_ds = H // h_gcd # downsample factor
    h_us = size // h_gcd # upsample factor
    w_ds = W // w_gcd
    w_us = size // w_gcd

    conv_L_us, conv_L_ds, conv_M_us, conv_M_ds = None, None, None, None
    if h_us in conv_mxs: # if precomputed matrix exists, call from memory (assuming KH==KW for simplicity)
        conv_L_us = conv_mxs[h_us]
    if h_ds in conv_mxs:
        conv_L_ds = conv_mxs[h_ds]
    if w_us in conv_mxs:
        conv_M_us = conv_mxs[w_us]
    if w_ds in conv_mxs:
        conv_M_ds = conv_mxs[w_ds]

    # Upsample and then downsample (Other order may lose too much information)
    upsampled_coeff, conv_L_us, conv_M_us = upsample_dct(coeff, L=h_us, M=w_us, dtype=dtype, conv_L=conv_L_us, conv_M=conv_M_us)
    resized_coeff, conv_L_ds, conv_M_ds = downsample_dct(upsampled_coeff, L=h_ds, dtype=dtype, M=w_ds, conv_L=conv_L_ds, conv_M=conv_M_ds)

    # assign conversion matrices to dict of pre-computed conversion matrices
    if conv_L_us != None and h_us not in conv_mxs:
        conv_mxs[h_us] = conv_L_us
    if conv_L_ds != None and h_ds not in conv_mxs:
        conv_mxs[h_ds] = conv_L_ds
    if conv_M_us != None and w_us not in conv_mxs:
        conv_mxs[w_us] = conv_M_us
    if conv_M_ds != None and w_ds not in conv_mxs:
        conv_mxs[w_ds] = conv_M_ds
    
    if dtype_out=='keep':
        return torch.round(resized_coeff).to(coeff_dtype)
    else:
        return resized_coeff.to(dtype_out)

## Resize related ops end

def crop_dct(coeff: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    """
    Cropping for DCT coefficients
    coeff: DCT coefficient with (C, H, W, KH, KW) structure where KH, KW = kernel height/width
    """

    c, h, w, kh, kw = coeff.shape
    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)]
        coeff = einops.rearrange(coeff, "c h w kh kw -> (c kh kw) h w")
        coeff = F.pad(coeff[..., max(top, 0) : bottom, max(left, 0) : right], padding_ltrb, fill=0)
        return einops.rearrange(coeff, "(c kh kw) hc wc -> c hc wc kh kw", c=c, kh=kh, kw=kw)
    return coeff[:, top:bottom, left:right, :, :]

def flip_dct(coeff: torch.Tensor, direction='horizontal', fixed_pos=False):
    """
    Flips dct coefficients to specified direction

    Args:
        coeff: DCT coefficient with 5 dimensions (c, h, w, kh, kw)
        direction: 'horizontal' or 'vertical' flip direction
        fixed_pos: don't flip block positions, just perform flip on individual blocks (used during rotation)
    Returns:
        Tensor: Flipped DCT coefficient
    """
    coeff = coeff.clone()
    if direction == 'horizontal':
        if not fixed_pos:
            coeff = coeff.flip(dims=(2,)) # flip block positions
        coeff[:,:,:,:,1::2] *= -1 # invert odd columns (odd-indexed columns when 0-indexed)
    elif direction == 'vertical':
        if not fixed_pos:
            coeff = coeff.flip(dims=(1,)) # flip block positions
        coeff[:,:,:,1::2,:] *= -1 # invert odd rows
    return coeff

def invert_dct(coeff: torch.Tensor):
    """
    Inverts 0-centered DCT coefficients

    Since the coefficients are zero centered, we can just multiply it by -1
    """
    return coeff * -1

def solarize_dct(coeff: torch.Tensor, threshold=0, dcThresholdMask=None):
    """
    Perform solarize augmentation on DCT coefficients

    Solarize the whole block where its DC component (DCT[0,0]) is above threshold

    Args:
        coeff: DCT coefficient (C, H, W, KH, KW)
        threshold: solarization threshold
        dcThresholdMask: Mask of indices to invert. If given, use this mask instead
    """
    assert len(coeff.shape) == 5, "Solarize only implemented for 5-channel DCT coeff (C, H, W, KH, KW)"
    coeff = coeff.clone()

    dcBlocks = coeff[:,:,:,0,0] # DC coefficients for all luma channel / single input channel
    if dcThresholdMask is None:
        dcThresholdMask = dcBlocks > threshold # dc blocks above threshold mask

    coeff[dcThresholdMask] *= -1 # invert all DCT coefficients if DC component is above threshold

    return coeff, dcThresholdMask

def solarize_add_dct(coeff: torch.Tensor, addition=128, threshold=0, dcThresholdMask=None):
    """ 
    Perform solarize_add augmentation on DCT coefficients

    If original DC component is below threshold, add 'addition' to it

    It makes more sense to use it only on Y

    Args:
        coeff: DCT coefficient (C, H, W, KH, KW)
        addition: Value to add to DC component for solarize decision
        threshold: solarization threshold
        dcThresholdMask: Mask of indices to invert. If given, use this mask instead
    """
    assert len(coeff.shape) == 5, "Solarize_add only implemented for 5-channel DCT coeff (C, H, W, KH, KW)"
    coeff = coeff.clone()

    dcBlocks = coeff[:,:,:,0,0] # DC coefficients for all luma channel / single input channel
    if dcThresholdMask is None:
        dcThresholdMask = dcBlocks < threshold # dc blocks below threshold mask

    dcBlocks[dcThresholdMask] += addition # add `addition` to DC component
    coeff[:,:,:,0,0] = dcBlocks # replace DC component

    coeff = torch.clamp(coeff, -2**10, 2**10-8) # clamp

    return coeff, dcThresholdMask

def sharpblur_dct(coeff: torch.Tensor, intensity, min=-2**10, max=2**10-8):
    """
    Sharpen/Blurs DCT coefficients by altering high-frequency components

    Args:
        coeff: DCT coefficient (C, H, W, KH, KW)
        intensity: Intensity of degradation. [-1, 1] where [-1,0):blur, [0,1]:sharpen
        min/max: min/max value to clamp coeff to after augmentation
    """
    assert len(coeff.shape) == 5, "Currently sharpblur_dct is only implemented for dct coeff of 5-dims (C, H, W, KH, KW)"
    assert intensity >= -1 and intensity <= 1, "Intensity should be within the range of [-1, 1]"
    coeff = coeff.clone()
    coeff_dtype = coeff.dtype
    _, _, _, KH, KW = coeff.shape

    filter_H = torch.linspace(1, (1+2*intensity), KH, dtype=torch.float32).unsqueeze(dim=1).clamp(min=0) # gradual frequency modification
    filter_W = torch.linspace(1, (1+2*intensity), KW, dtype=torch.float32).unsqueeze(dim=0).clamp(min=0)
    filterMat = filter_H.mm(filter_W) # KH, KW
    filterMat = filterMat

    filterMat = filterMat.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0) # 1, 1, 1, KH, KW
    coeff = coeff * filterMat # apply linear filter
    coeff = coeff.clamp(min=min, max=max) # clamp min/max
    
    if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        coeff = torch.round(coeff)

    return coeff.to(coeff_dtype)

def midfreqaug_dct(coeff: torch.Tensor, intensity, min=-2**10, max=2**10-8):
    """
    Sharpen/Blurs DCT coefficients by altering mid-frequency components (lower higher-frequency alterations)

    Args:
        coeff: DCT coefficient (C, H, W, KH, KW)
        intensity: Intensity of degradation. [-1, 1] where [-1,0):blur, [0,1]:sharpen
        min/max: min/max value to clamp coeff to after augmentation
    """
    assert len(coeff.shape) == 5, "Currently midfreqblur_dct is only implemented for dct coeff of 5-dims (C, H, W, KH, KW)"
    assert intensity >= -1 and intensity <= 1, "Intensity should be within the range of [-1, 1]"
    coeff = coeff.clone()
    coeff_dtype = coeff.dtype
    _, _, _, KH, KW = coeff.shape

    coeff = blockshift(coeff, dim=(-2, -1)) # blockshift so that DC component is moved to the center

    # Create gaussian low-pass filter
    H_intensity = KH//2 - (KH//8 * 2.2) * abs(intensity) # 1.8 * KH//8 ~ 4 * KH//8
    W_intensity = KW//2 - (KW//8 * 2.2) * abs(intensity) # 1.8 * KW//8 ~ 4 * KW//8

    ## Using gaussian or linear window (as in `sharpblur_dct`) is a design choice and can be interchanged when necessary
    filter_H = torch.tensor(scipy.signal.windows.gaussian(KH, H_intensity), dtype=torch.float32).unsqueeze(dim=1) # KH, 1 # generate 2x the window (circular) and crop the fourth quadrant (top left = one, top bottom = smallest)
    filter_W = torch.tensor(scipy.signal.windows.gaussian(KW, W_intensity), dtype=torch.float32).unsqueeze(dim=0) # 1, KW (~1ms)
    filterMat = filter_H.mm(filter_W) # KH, KW

    if intensity >= 0:
        filterMat = 1/filterMat # Invert filter
    filterMat = filterMat.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0) # 1, 1, 1, KH, KW
    coeff = coeff * filterMat # apply gaussian filter
    coeff = coeff.clamp(min=min, max=max) # clamp min/max
    coeff = iblockshift(coeff, dim=(-2, -1))
    
    if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        coeff = torch.round(coeff)

    return coeff.to(coeff_dtype)

def translate_dct(coeff: torch.Tensor, magnitude, direction="H"):
    """
    Translates DCT coefficients by magnitude

    Args:
        coeff: DCT coefficient (C, H, W, KH, KW)
        magnitude: magnitude of translation (moves by "magnitude" blocks) (can be positive or negative)
        direction: "H" or "W" -- height or width
    """
    assert len(coeff.shape) == 5, "Currently translate is only implemented for DCT coeff of dim=5"
    _, H, W, _, _ = coeff.shape
    coeff = coeff.clone()
    if direction=='H':
        assert abs(magnitude) < H, "You cannot translate more than the image's height"
        coeff = torch.roll(coeff, magnitude, dims=(1,))
        if magnitude >= 0:
            coeff[:, :magnitude, :, :, :] = 0 # 0~magnitude = 0
        else:
            coeff[:, magnitude:, :, :, :] = 0 # -magnitude~0 = 0
    elif direction=="W":
        assert abs(magnitude) < W, "You cannot translate more than the image's width"
        coeff = torch.roll(coeff, magnitude, dims=(2,))
        if magnitude >= 0:
            coeff[:, :, :magnitude, :, :] = 0 # 0~magnitude = 0
        else:
            coeff[:, :, magnitude:, :, :] = 0 # -magnitude~0 = 0
    return coeff

def cutout_dct(
    coeff: torch.Tensor, pad_size, replace=0,
    cutout_center_height=None, cutout_center_width=None):
    """
    Randomly replaces (2*pad_size, 2*pad_size) patch in the image to `replace`
    
    Args:
        coeff: DCT Coeff of shape (C,H,W,KH,KW)
        pad_size: size of replace mask (mask will be size (2*pad_size, 2*pad_size))
        replace: value to replace the mask with
        cutout_center_height, cutout_center_width: if given, use this center instead
    """
    assert len(coeff.shape) == 5, "Currently cutout is only implemented for DCT coeff of 5 dims"
    coeff = coeff.clone()
    C, H, W, _, _ = coeff.shape
    if cutout_center_height is None:
        cutout_center_height = (torch.randint(low=0, high=H, size=(1,)).item()) // 2 * 2
    if cutout_center_width is None:
        cutout_center_width = (torch.randint(low=0, high=W, size=(1,)).item()) // 2 * 2
    
    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = max(0, H - cutout_center_height - pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = max(0, W - cutout_center_width - pad_size)

    cutout_shape = (H - (lower_pad + upper_pad),
                    W - (left_pad + right_pad)) # cutout this shape
    padding_dims = (left_pad, right_pad, upper_pad, lower_pad) # (W, H) paddings
    cutout_mask = torch.nn.functional.pad(
        torch.zeros(cutout_shape, dtype=coeff.dtype, device=coeff.device),
        padding_dims, value=1
    ) # (H, W)
    cutout_mask = cutout_mask.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) # 1, H, W, 1, 1
    cutout_mask = torch.tile(cutout_mask, (C, 1, 1, 1, 1))
    coeff = torch.where(
        cutout_mask==0, # condition
        torch.ones_like(coeff) * replace, # If True
        coeff # if False
    )
    return coeff, cutout_center_height, cutout_center_width

def brightness_dct(
    coeff:torch.Tensor, factor:float):
    """
    Adjusts brightness of a DCT coefficient (only use on Y)

    Args:
        coeff (Tensor): DCT coefficient of shape (C,H,W,KH,KW)
        factor: brightness adjustment factor in [0,inf], 0: 2x darker image, 1: No change, 2: 2x brighter image
    """
    # Adjust brightness by manipulating DC component of a DCT coefficient (DCT[0,0])
    assert len(coeff.shape)==5, "Currently brightness_dct is only supported for 5-dim DCT coeff"
    assert factor >= 0, "Brightness adjustment factor should be non-negative"
    coeff = coeff.clone()
    coeff_dtype = coeff.dtype
    coeff_dc = coeff[:,:,:,0,0].to(torch.float32)
    coeff_dc += torch.mean(torch.abs(coeff_dc)) * (factor - 1) # no blockiness
    
    if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        coeff_dc = torch.round(coeff_dc)
    coeff[:,:,:,0,0] = coeff_dc.to(coeff_dtype)
    return coeff

def contrast_dct(
    coeff:torch.Tensor, factor:float):
    """
    Adjusts contrast of a DCT coefficient (If used on Y)
    If this is used on cbcr (chroma channel) it will work as adjusting color saturation instead

    Args:
        coeff (Tensor): DCT coefficient of shape (C,H,W,KH,KW)
        factor: contrast adjustment factor in [0,3], 0: 2x less contrast image, 1: No change, 2: 2x contrast image
    """
    # Adjust brightness by manipulating DC component of a DCT coefficient (DCT[0,0])
    assert len(coeff.shape)==5, "Currently contrast_dct is only supported for 5-dim DCT coeff"
    assert factor >= 0 and factor <= 3, "Contrast adjustment factor should be in range [0,3]"
    coeff = coeff.clone()
    coeff_dtype = coeff.dtype
    coeff_dc = coeff[:,:,:,0,0].to(torch.float32)
    
    coeff_dc *= factor
    if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        coeff_dc = torch.round(coeff_dc)
    coeff[:,:,:,0,0] = coeff_dc.to(coeff_dtype)
    return coeff

def autocontrast_dct(coeff:torch.Tensor, min=-2**10, max=2**10-8):
    """
    Auto-contrast DCT coefficient so that the darkest patch gets scaled to black and brightest patch gets scaled to white
    Only use on Y (don't use on cbcr)

    Args:
        coeff (Tensor): DCT coefficient of shape (C, H, W, KH, KW) (should be de-quantized DCT in order to use default range)
        min: minimum value of DCT coefficient (Default: -2**10) -- only modify if you are training on non-8-bit image
        max: maximum value of DCT coefficient (Default: 2**10-8) -- only modify if you are training on non-8-bit image
    """
    assert len(coeff.shape)==5, "Currently autocontrast_dct is only supported for 5-dim DCT coeff"
    coeff = coeff.clone()
    coeff_dtype = coeff.dtype
    coeff_dc = coeff[:,:,:,0,0].to(torch.float32)
    coeff_dc_min = coeff_dc.min()
    coeff_dc_max = coeff_dc.max()
    
    if coeff_dc_min == coeff_dc_max and coeff_dc_max==0: # if both zero, do not scale
        return coeff
    coeff_dc = (coeff_dc - coeff_dc_min) / (coeff_dc_max - coeff_dc_min) # normalize to 0~1
    coeff_dc = min + (coeff_dc * (max - min)) # scale to min~max

    if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        coeff_dc = torch.round(coeff_dc)
    coeff[:,:,:,0,0] = coeff_dc.to(coeff_dtype)
    return coeff

def posterize_dct(coeff:torch.Tensor, bitoffset=2, min=-2**10, max=2**10-8):
    """
    Posterize DCT coefficient by `bitoffset`. i.e. remove `bitoffset` number of bits (quantize)

    Args:
        coeff (Tensor): DCT coefficient of shape (C,H,W,KH,KW)
        bitoffset: number of bits to subtract
        min: minimum value of DCT coefficient (Default: -2**10) -- only modify if you are training on non-8-bit image
        max: maximum value of DCT coefficient (Default: 2**10-8) -- only modify if you are training on non-8-bit image
    """
    assert len(coeff.shape)==5, "Currently posterize_dct is only supported for 5-dim DCT coeff"
    coeff = coeff.clone()
    coeff_dtype = coeff.dtype
    coeff_dc = coeff[:,:,:,0,0].to(torch.float32)
    coeff_dc = coeff_dc - min # 0~(max-min)
    assert coeff_dc.min()>=0, f"coeff_dc should be positive after subtracting min. Current min:{coeff_dc.min()}, max: {coeff_dc.max()}"
    coeff_dc /= 2**bitoffset
    coeff_dc = torch.round(coeff_dc) # index
    coeff_dc = coeff_dc.to(torch.int64)
    quant_table = torch.linspace(min, max, round((max-min)/(2**bitoffset))+1) # has to use round for edge cases (ex 1023 / 128 = 7.99 but should be counted as 8)
    coeff_dc = quant_table[coeff_dc]

    if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        coeff_dc = torch.round(coeff_dc)
    coeff[:,:,:,0,0] = coeff_dc.to(coeff_dtype)
    return coeff

def scale_channel_dct(coeff_chan:torch.Tensor, min=-2**10, max=2**10-8):
    """
    Scale DC component of DCT coefficient w.r.t. histogram

    Args:
        coeff_chan: DC component without channel (H, W, KH, KW)
        min: minimum value of DCT coefficient (Default: -2**10) -- only modify if you are training on non-8-bit image
        max: maximum value of DCT coefficient (Default: 2**10-8) -- only modify if you are training on non-8-bit image
    """
    assert len(coeff_chan.shape) == 4, "coeff_chan should not have a channel dimension; should be (H, W, KH, KW)"
    
    coeff_chan_dc = coeff_chan[:,:,0,0].clone() - min # rearrange to 0~(max-min)
    assert coeff_chan_dc.min() >=0, f"coeff_chan_dc should be positive after subtracting min. Current min:{coeff_chan_dc.min()}, max: {coeff_chan_dc.max()}"
    coeff_dtype = coeff_chan.dtype
    if coeff_chan_dc.is_cuda:
        hist = torch.histc(coeff_chan_dc.to(torch.float32), bins=2048, min=0, max=(max-min))
    else:
        hist = torch.bincount(coeff_chan_dc.view(-1), minlength=(max-min)+1)
    
    nonzero_hist = hist[hist != 0]
    mn_minus_min = nonzero_hist[1:].sum() # num_pixels (or coefficients in this case) - cdf_min
    cdf = torch.cumsum(hist, 0) # cumulative density function
    equalized = torch.round((cdf-nonzero_hist[0]) / mn_minus_min * (max-min-1)) # (cdf - cdf_min) / (num_pixels - cdf_min) * (num_total_grayscale_range - 1)
    coeff_chan_dc = equalized[coeff_chan_dc.to(torch.int64)].to(coeff_dtype) + min # equalize dc components
    coeff_chan[:,:,0,0] = coeff_chan_dc
    return coeff_chan.to(coeff_dtype)

def equalize_dct(coeff:torch.Tensor, min=-2**10, max=2**10-8):
    """
    Perform histogram equalization for DCT coefficient
    Equalizes DC component of each blocks
    Only use on Y (don't use on cbcr)

    Args:
        coeff (Tensor): DCT coefficient of shape (C, H, W, KH, KW)
    """
    assert len(coeff.shape)==5, "Currently equalization is only supported for 5-dim DCT coeff"
    coeff = coeff.clone()
    C, _, _, _, _ = coeff.shape
    return torch.stack([scale_channel_dct(coeff[c], min=min, max=max) for c in range(C)])
        
def shear_block(coeff: torch.Tensor, deg_x=0, deg_y=0, conv_L=None, conv_M=None, dtype='keep', window=False, pad=False):
    """
    Shears DCT coefficient by degrees counter-clockwise

    Algorithm: 
        1. Combine blockwise DCT coeff (C, H, W, KH, KW) -> C, (H KH), (W, KW)
        2. Rotate combined DCT coeff using dct2dft rotation method
        3. Decompose combined DCT coeff (C, (H KH), (W KW)) to (C, H, W, KH, KW)

    Input:
        coeff: DCT coefficient (C, H, W, KH, KW)
        deg_x, deg_y: Magnitude of shearing in x and y dimension (in degrees)
        conv_L: conversion matrix for height (L, H -> (L*H))
        conv_M: conversion matrix for width (W, M -> (W*M))
        dtype: if 'keep', keep coeff dtype. else, cast to dtype
        window: Use tukey window per block if true
        pad: NEW: Size of padding (if false, no padding). Ex: 3 = 1 pad, 1 img, 1 pad on each sides. OLD: If true, pads image to (sqrt(2)*(H, W))
    """
    C, H, W, KH, KW = coeff.shape
    coeff_dtype = coeff.dtype

    if pad: 
        assert pad >= 1, "Padding should be larger than 1"
        coeff_padded = torch.zeros((C, int(H*pad//1), int(W*pad//1), KH, KW), dtype=coeff.dtype, device=coeff.device) # padding
        Hmargin = (int(H*pad//1)-H)//2
        Wmargin = (int(W*pad//1)-W)//2
        coeff_padded[:, Hmargin:Hmargin+H, Wmargin:Wmargin+W, :, :] = coeff
    else:
        coeff_padded=coeff

    _, Hp, Wp, _, _= coeff_padded.shape
    
    if window: # tukey alpha=0: rectangular window, alpha=1 = Hann window
        window_H = torch.tensor(scipy.signal.tukey(Hp, alpha=0.4), dtype=torch.float32).unsqueeze(dim=1)
        window_W = torch.tensor(scipy.signal.tukey(Wp, alpha=0.4), dtype=torch.float32).unsqueeze(dim=0)
        window_mat = window_H.mm(window_W)
        coeff_padded = coeff_padded.to(torch.float32) * window_mat.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) # element-wise multiplication

    coeff_padded = blockshift(coeff_padded, dim=(1, 2))
    
    coeff_composed, conv_L, conv_M = combine_blocks_dft(coeff_padded, conv_L, conv_M)
    coeff_dft_sheared = dtu.shear_dft_2d_spatial(coeff_composed, deg_x, deg_y)
    coeff_decomposed, _, _ = decompose_block_dft(coeff_dft_sheared, Hp, Wp, KH, KW, conv_L, conv_M)
    
    coeff_decomposed = iblockshift(coeff_decomposed, dim=(1, 2))

    if pad: #unpad
        coeff_decomposed = coeff_decomposed[:, Hmargin:Hmargin+H, Wmargin:Wmargin+W, :, :]
    
    if dtype=='keep':
        if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            coeff_decomposed = torch.round(coeff_decomposed)
        return coeff_decomposed.to(coeff_dtype), conv_L, conv_M
    else:
        if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            coeff_decomposed = torch.round(coeff_decomposed)
        return coeff_decomposed.to(dtype), conv_L, conv_M

def freq_enhance_dct(coeff:torch.Tensor, magnitude=1):
    """
    Enhances frequency component of a DCT coefficient

    Args:
        coeff (Tensor): DCT coefficient of shape (C, H, W, KH, KW)
        magnitude: Multiply frequency component by this amount
    """
    assert len(coeff.shape)==5, "Currently equalization is only supported for 5-dim DCT coeff"
    coeff = coeff.clone()
    _, _, _, KH, KW = coeff.shape
    coeff_dtype=coeff.dtype
    coeff = coeff.to(torch.float32)

    coeff = einops.rearrange(coeff, "c h w kh kw -> c h w (kh kw)")
    coeff[:, :, :, 1:] *= magnitude # multiply frequency (anything except dct[0,0] by `magnitude`)
    coeff = einops.rearrange(coeff, "c h w (kh kw) -> c h w kh kw", kh=KH, kw=KW)

    if coeff_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        coeff = torch.round(coeff)
    return coeff.to(coeff_dtype)