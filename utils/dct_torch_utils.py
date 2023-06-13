"""
DCT utils for pytorch
"""
import torch
import torchvision.transforms.functional as F_tv
import utils.dct_ops as dops

def fct_1d(signal, norm='ortho'):
    """
    Fast DCT by J. Makhoul -- A fast cosine transform in one and two dimensions
    Computes Fast DCT on the last dimension of the input signal
    
    signal: input signal (torch tensor)
    norm: 'ortho' or None
    """
    sig_shape = torch.tensor(signal.shape)
    dimlen = sig_shape[-1]
    sig_flat = torch.flatten(signal, start_dim=0, end_dim=-2)

    sig_rearrange = torch.cat(
        (sig_flat[:, ::2], sig_flat[:,1::2].flip(dims=(1,))), dim=1
    )

    FCT_coeff = torch.fft.fft(sig_rearrange, dim=-1)

    exp_coeff = -1j*torch.arange(dimlen, dtype=torch.int64, device=signal.device) * torch.pi / (2 * dimlen)
    W = exp_coeff.exp()
    
    FCT_coeff = FCT_coeff * W

    if norm=='ortho':
        FCT_coeff[:, 0] /= torch.sqrt(dimlen) * 2
        FCT_coeff[:, 1:] /= torch.sqrt(dimlen / 2) * 2
    FCT_coeff = 2 * FCT_coeff.view(signal.shape)

    return FCT_coeff.real

def ifct_1d(FCT_coeff, norm='ortho'):
    """
    FCT_coeff: DCT coefficient
    norm: 'ortho' for orthogonal normalization or None
    """
    coeff_shape = torch.tensor(FCT_coeff.shape)
    dimlen = coeff_shape[-1]
    coeff_flat = torch.flatten(FCT_coeff, start_dim=0, end_dim=-2).clone()

    if norm=='ortho':
        coeff_flat[:,0] *= torch.sqrt(dimlen) * 2
        coeff_flat[:, 1:] *= torch.sqrt(dimlen/2) * 2

    coeff_flat_flip = torch.cat((coeff_flat[:,:1]*0, coeff_flat[:,1:].flip(dims=(-1,))), dim=-1)

    exp_coeff = 1j*torch.arange(dimlen, dtype=torch.int64, device=FCT_coeff.device) * torch.pi / (2 * dimlen)
    W = exp_coeff.exp()

    dft_coeff = W * (coeff_flat - 1j*coeff_flat_flip) / 2 # YESSS

    signal_rearr = torch.fft.ifft(dft_coeff, dim=-1).real
    signal_flat = torch.zeros_like(signal_rearr)
    signal_flat[:, ::2] = signal_rearr[:, :dimlen.item() - (dimlen.item() // 2)]
    signal_flat[:, 1::2] = signal_rearr[:, dimlen.item() - (dimlen.item() // 2):].flip(dims=(-1,))
    signal = signal_flat.view(FCT_coeff.shape)

    return signal

def fct_2d(signal, norm='ortho'):
    """
    performs Fast DCT on last two dimensions of the input signal
    
    signal: input signal (.... H, W)
    norm: 'ortho' or None
    """
    FCT1 = fct_1d(signal, norm=norm)
    FCT2 = fct_1d(FCT1.transpose(-1, -2), norm=norm)
    return FCT2.transpose(-1, -2)

def ifct_2d(FCT_coeff, norm='ortho'):
    """
    Performs Fast inverse DCT on last to dimensions of the input coefficients

    FCT_coeff: 2-D DCT Coefficients
    norm: 'ortho' or None
    """
    signal1 = ifct_1d(FCT_coeff, norm=norm)
    signal = ifct_1d(signal1.transpose(-1,-2),norm=norm)
    return signal.transpose(-1,-2)

def dct2dft_fast_1d(dctcoeff, norm='ortho'):
    """
    1-D transform of input DCT coefficients to DFT coefficients
    """
    coeff_shape = torch.tensor(dctcoeff.shape)
    dimlen = coeff_shape[-1]
    coeff_flat = torch.flatten(dctcoeff, start_dim=0, end_dim=-2).clone()

    if norm=='ortho':
        coeff_flat[:,0] *= torch.sqrt(dimlen) * 2
        coeff_flat[:, 1:] *= torch.sqrt(dimlen/2) * 2

    coeff_flat_flip = torch.cat((coeff_flat[:,:1]*0, coeff_flat[:,1:].flip(dims=(-1,))), dim=-1)
    exp_coeff = 1j*torch.arange(dimlen, dtype=torch.int64, device=dctcoeff.device) * torch.pi / (2 * dimlen)
    W = exp_coeff.exp()
    dft_coeff = W * (coeff_flat - 1j*coeff_flat_flip) / 2

    return dft_coeff.view(dctcoeff.shape)

def dct2dft_fast_2d(dctcoeff, norm='ortho'):
    """
    2-D conversion of input DCT coefficients to DFT
    """
    DFT1 = dct2dft_fast_1d(dctcoeff, norm=norm)
    DFT2 = dct2dft_fast_1d(DFT1.transpose(-1, -2), norm=norm)
    return DFT2.transpose(-1,-2)

def dft2dct_1d(dftcoeff, norm='ortho'):
    """
    1-D conversion of input DFT coefficient to DCT
    """
    dftcoeff_shape = torch.tensor(dftcoeff.shape)
    dimlen = dftcoeff_shape[-1]
    dftcoeff_flat = torch.flatten(dftcoeff, start_dim=0, end_dim=-2).clone()

    exp_coeff = -1j*torch.arange(dimlen, dtype=torch.int64, device=dftcoeff.device) * torch.pi / (2 * dimlen)
    W = exp_coeff.exp()

    dct_complex_coeff = 2 * dftcoeff_flat * W
    dct_coeff = dct_complex_coeff.real

    if norm=='ortho':
        dct_coeff[:,0] /= torch.sqrt(dimlen) * 2
        dct_coeff[:, 1:] /= torch.sqrt(dimlen/2) * 2
    
    return dct_coeff.view(dftcoeff.shape)

def dft2dct_2d(dftcoeff, norm='ortho'):
    """
    2-D conversion of input DFT coefficients to DCT coefficients
    """
    DCT1 = dft2dct_1d(dftcoeff, norm=norm)
    DCT2 = dft2dct_1d(DCT1.transpose(-1, -2), norm=norm)
    return DCT2.transpose(-1,-2)

def dct2dft_2n_1d(sequence, norm='ortho'):
    """
    recovers fft of mirrored sequence (2N -- [a b c d d c b a] instead of N -- [a d c b]) from dct
    """
    sequence_output_shape = list(sequence.shape) # original shape
    sequence_output_shape[-1] *= 2 # expand last dimension by 2
    sequence = sequence.clone()
    sequence = torch.flatten(sequence, start_dim=0, end_dim=-2)
    N,S = sequence.shape # flattened shape
    reconst_seq = torch.zeros((N, S*2), dtype=torch.complex64)
    exp_coeff = torch.arange(S*2) * (1j) * torch.pi / (2*S) # notice the lack of sign in front of 1j
    
    if norm=='ortho':
        # recovers non-orthogonal dct
        sequence[:,0] *= (S**0.5) * 2
        sequence[:, 1:] *= ((S/2)**0.5) * 2
    
    reconst_seq[:,:S] = sequence
    reconst_seq[:,S+1:] = -sequence.flip(dims=(-1,))[:,:-1] # there must be 0 in the middle
    
    return (reconst_seq * exp_coeff.exp()).view(sequence_output_shape)

def dft2dct_2n_1d(dftseq, norm='ortho', takereal=True):
    """
    recovers DCT coefficient from mirrored dft (2N -> N transform)
    
    takereal: whether or not to take the real values only at the output (always true if using 1d decomposition, for 2d, only true on the last decomposition)
    """
    dftseq_output_shape = list(dftseq.shape) # original shape
    dftseq_output_shape[-1] //= 2 # reduce last dimension by 2
    dftseq = dftseq.clone()
    dftseq = torch.flatten(dftseq, start_dim=0, end_dim=-2)
    if dftseq.dtype not in [torch.complex64, torch.complex128]:
        dftseq = dftseq.to(torch.complex64) # cast to complex if not complex
    N, S2 = dftseq.shape # flattened shape
    dftseq = dftseq.clone()
    exp_coeff = torch.arange(S2) * (-1j) * torch.pi / (S2) # notice the sign in front of 1j
    dftseq *= exp_coeff.exp()

    if norm=='ortho':
        # recovers non-orthogonal dct
        dftseq[:,0] /= ((S2//2)**0.5) * 2
        dftseq[:, 1:] /= ((S2/4)**0.5) * 2
    
    if takereal:
        dctseq = dftseq[:,:(S2//2)].real
    else:
        dctseq = dftseq[:,:(S2//2)]

    return dctseq.view(dftseq_output_shape)

def dct2dft_2n_2d(dctcoeff, norm='ortho'):
    """
    2-D conversion of input DCT (N) coefficients to DFT (2N)
    """
    DFT1 = dct2dft_2n_1d(dctcoeff, norm=norm)
    DFT2 = dct2dft_2n_1d(DFT1.transpose(-1, -2), norm=norm)
    return DFT2.transpose(-1,-2)

def dft2dct_2n_2d(dftcoeff, norm='ortho'):
    """
    2-D conversion of input DFT (2N) coefficients to DCT (N) coefficients
    """
    DCT1 = dft2dct_2n_1d(dftcoeff, norm=norm, takereal=False)
    DCT2 = dft2dct_2n_1d(DCT1.transpose(-1, -2), norm=norm, takereal=True)
    return DCT2.transpose(-1,-2)

def phase_shift_dft_1d(dftcoeff, degrees=45):
    """
    Phase shifts the last dimension by 'degrees' (in imaginary plane) (1D)
    """
    degrees_rad = degrees/180*torch.pi
    coeff_shape = torch.tensor(dftcoeff.shape)
    dimlen = coeff_shape[-1]
    coeff_flat = torch.flatten(dftcoeff, start_dim=0, end_dim=-2).clone()
    exp_coeff = 1j*torch.ones(dimlen, dtype=torch.int64, device=dftcoeff.device) * degrees_rad
    EXP = exp_coeff.exp()
    
    rotated_dftcoeff = EXP * coeff_flat
    return rotated_dftcoeff.view(dftcoeff.shape)

def phase_shift_dft_2d(dftcoeff, degrees=45):
    """
    Phase shifts the last two dimension by 'degrees' (in imaginary plane) (2D)
    """
    DFT1 = phase_shift_dft_1d(dftcoeff, degrees=degrees)
    DFT2 = phase_shift_dft_1d(DFT1.transpose(-1, -2), degrees=degrees)
    return DFT2.transpose(-1,-2)

def rotate_dft_2d_spatial(dftcoeff, degrees=45, real=False, use_blockshift=False):
    """
    Spatial rotation of the last two dimensions
    1. FFTshift -> centers FFT coeffieints
    2. Flatten -> easier to handle rotation
    3. Rotate -> rotate real and imaginary parts together (counter-clockwise)
    4. Reshape to original shape
    5. Inverse FFTshift to retrieve the rotated coefficients

    input
    dftcoeff: DFT coefficient to rotate
    degrees: degrees to rotate
    real: If True, only rotate real part (used for DCT rotation)
    use_blockshift: If true, use blockshift instead of fftshift (no meaningful difference in performance)
    """
    degrees *= -1 # this makes rotation counter-clockwise (in agreement with most rotation methods)
    if use_blockshift:
        dftcoeff_shifted = dops.blockshift(dftcoeff, dim=(-2, -1))
    else:
        dftcoeff_shifted = torch.fft.fftshift(dftcoeff, dim=(-2, -1))
    dftcoeff_shiftflatten = torch.flatten(dftcoeff_shifted, start_dim=0, end_dim=-3)
    _, H, W = dftcoeff_shiftflatten.shape
    if not real:
        dftcoeff_rotated = F_tv.rotate(dftcoeff_shiftflatten.real, degrees, expand=False) + \
            1j*F_tv.rotate(dftcoeff_shiftflatten.imag, degrees, expand=False)
    else:
        dftcoeff_rotated = F_tv.rotate(dftcoeff_shiftflatten, degrees, expand=False)
    dftcoeff_rotated = dftcoeff_rotated.view(dftcoeff_shifted.shape)
    if use_blockshift:
        dftcoeff_unshift = dops.iblockshift(dftcoeff_rotated, dim=(-2,-1))
    else:
        dftcoeff_unshift = torch.fft.ifftshift(dftcoeff_rotated, dim=(-2, -1))

    return dftcoeff_unshift

def shear_dft_2d_spatial(dftcoeff, deg_x=0, deg_y=0, real=False, use_blockshift=False):
    """
    Shear last two dimensions of dft coefficient
    1. FFTshift -> centers FFT coeffieints
    2. Flatten -> easier to handle rotation
    3. Shear -> shear real and imaginary parts together
    4. Reshape to original shape
    5. Inverse FFTshift to retrieve the rotated coefficients

    input
    dftcoeff: DFT coefficient to rotate
    deg_x, deg_y: Degree of shearing in x and y dimension (in degrees of course)
    real: If True, only rotate real part (used for DCT rotation)
    use_blockshift: If true, use blockshift instead of fftshift (not useful in practice)
    """
    #degrees *= -1 # this makes rotation counter-clockwise (in agreement with most rotation methods)
    if use_blockshift:
        dftcoeff_shifted = dops.blockshift(dftcoeff, dim=(-2, -1))
    else:
        dftcoeff_shifted = torch.fft.fftshift(dftcoeff, dim=(-2, -1))
    dftcoeff_shiftflatten = torch.flatten(dftcoeff_shifted, start_dim=0, end_dim=-3)
    _, H, W = dftcoeff_shiftflatten.shape
    if not real:
        dftcoeff_sheared = F_tv.affine(
                dftcoeff_shiftflatten.real,
                angle=0.,
                translate=[0,0],
                scale=1.0,
                shear=[deg_x, deg_y],
                fill=0,
            ) + \
            1j*F_tv.affine(
                dftcoeff_shiftflatten.imag,
                angle=0.,
                translate=[0,0],
                scale=1.0,
                shear=[deg_x, deg_y],
                fill=0,
            )
    else:
        dftcoeff_sheared = F_tv.affine(
                dftcoeff_shiftflatten.real,
                angle=0.,
                translate=[0,0],
                scale=1.0,
                shear=[deg_x, deg_y],
                fill=0,
            )
    dftcoeff_sheared = dftcoeff_sheared.view(dftcoeff_shifted.shape)
    if use_blockshift:
        dftcoeff_unshift = dops.iblockshift(dftcoeff_sheared, dim=(-2,-1))
    else:
        dftcoeff_unshift = torch.fft.ifftshift(dftcoeff_sheared, dim=(-2, -1))

    return dftcoeff_unshift