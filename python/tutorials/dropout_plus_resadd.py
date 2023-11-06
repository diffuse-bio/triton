"""
Dropout --> resadd --> CLN (DRCLN)
"""


import time
import torch

import triton
import triton.language as tl
import torch.nn.functional as F

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False


#
# X --> [some operations] --> Z --> dropout --> Res add_x --> LN/CLN --. Y

@triton.jit
def _drcln_fwd_fused(
    X,  # pointer to the original input (that will be added -- res add)
    Y,  # pointer to the output
    Z, # pointer to the input to dropout
    YIN, # pointer to output after dropout and res add
    MASK_out, # pointer to output mask after dropout
    p, # dropout prob
    seed, # dropout seed
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    # RANDOM,
    BLOCK_SIZE: tl.constexpr,
):
    
    # x (input for res add)
    # z (input for dropout)
    # zout = dropout(z)
    # yin = zout + x
    # y = LN[W, B](yin)

    # Map the program id to the row of X, Y, and Z it should compute.
    row = tl.program_id(0)

    Z += row * stride
    Y += row * stride
    X += row * stride
    YIN += row * stride
    MASK_out += row * stride
    W += row * stride
    B += row * stride
    # RANDOM += row * stride

    # seed += row * seed_stride

    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        all_elem = row * stride + (off + tl.arange(0, BLOCK_SIZE))
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # do dropout on Z
        z = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        random = tl.rand(seed, all_elem)
        if False:
            tl.store(RANDOM + cols, random, mask=mask)
        x_keep = random > p
        # _mean += a
        z_out = tl.where(x_keep, z / (1 - p), 0.0)

        # residual add x 
        yin = x + z_out
        tl.store(MASK_out + cols, x_keep, mask=mask)
        tl.store(YIN + cols, yin, mask=mask)

        # get mean of output 
        # a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += yin

    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        yin = tl.load(YIN + cols, mask=cols < N, other=0.).to(tl.float32)
        yin = tl.where(cols < N, yin - mean, 0.)
        _var += yin * yin
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        yin = tl.load(YIN + cols, mask=cols < N, other=0.).to(tl.float32)
        # x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        yinhat = (yin - mean) * rstd
        # if False: # CHANGE BACK
        y = yinhat * w + b
        # Write output -- CHNAGE BACK TO Y
        tl.store(Y + cols, y, mask=mask)



@triton.jit
def _drcln_bwd_dx_fused(
    DY, # pointer to gradient wrt LN output
    DYIN, # pointer to gradient wrt residual
    DZ, # pointer to input gradient (input ahead of dropout)
    DX, # pointer input gradient (input for residual add),
    YIN, # input to CLN
    # Z, # pointer to input
    W, # CLN weights,
    DW,# pointer to output gradient for CLN scale
    DB, # pointer to output gradient for CLN shift
    Mean,
    Rstd,
    DROPOUT_MASK, # pointer to dropout mask
    p, # dropout rate

    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    # eps,  # epsilon to avoid division by zero
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # x (input for res add)
    # z (input for dropout)
    # zout = dropout(z)
    # yin = zout + x
    # y = LN[W, B](yin)

    # X dw = dy * yin
    # X db = dy
    # dyin =(wdy - (yin_hat * c1 + c2)) * rstd
    # dx = dyin
    # dz = dyin * mask / (1 - p)


    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    DY += row * stride
    DYIN += row * stride
    YIN += row * stride
    W += row * stride
    DW += row * stride
    DB += row * stride
    DX += row * stride
    DROPOUT_MASK += row * stride
    DZ += row * stride

    # input to CLN
    yin = tl.load(YIN + cols, mask=mask, other=0).to(tl.float32)
    # gradient
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    dyin = tl.load(DYIN + cols, mask=mask, other=0).to(tl.float32)
    # CLN weight (scale)
    w = tl.load(W + cols, mask=mask).to(tl.float32)

    # get DW, DB
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    yin_hat = (yin - mean) * rstd
    wdy = w * dy
    yin_hat = tl.where(mask, yin_hat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(yin_hat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    # update gradient for input to CLN with dy/dCLN
    dyin += (wdy - (yin_hat * c1 + c2)) * rstd
    # Write dx ==  dx is just dyin (res add)
    tl.store(DX + cols, dyin, mask=mask)
    # Accumulate partial sums for dw/db
    # TESTING -- change back
    dw = (dy * yin_hat).to(w.dtype) 
    db = (dy).to(w.dtype) # / ( 1- p)
    
    tl.store(DW + cols, dw, mask=mask)
    tl.store(DB + cols, db, mask=mask)

    # compute dz (dropout then add)
    dropout_mask = tl.load(DROPOUT_MASK + cols, mask=mask, other=0.).to(tl.float32)
    dz = dyin * dropout_mask / (1 - p)
    tl.store(DZ + cols, dz, mask=mask)





# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.


class DropoutResAddCLN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, x, p, weight, bias, eps, seed): #: bool = False):
        # z --> input ahead of dropout
        # x --> original input (residual add)

        # allocate output
        B, num_res, d = x.size()
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        z_arg = z.reshape(-1, x.shape[-1])
        w_arg = weight.reshape(-1, x.shape[-1])
        b_arg = bias.reshape(-1, x.shape[-1])
        y = torch.empty_like(x)
        yin = torch.empty_like(z)
        mask_out = torch.empty_like(z)
        M, N = x_arg.shape

        mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel

        _drcln_fwd_fused[(M,)](x_arg, y, z_arg, yin, mask_out, p, seed, w_arg, b_arg, mean, rstd,
                                    x_arg.stride(0), N, eps, 
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(mask_out, mean, rstd, weight, yin ) 
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.p = p
        ctx.B = B
        ctx.num_res = num_res
        y = y.reshape(B, num_res, d)
        residual = yin.reshape(B, num_res, d)
        return y, residual, mask_out

    @staticmethod
    def backward(ctx, dy, dresidual, dmask_out):
        dropout_mask, mean, rstd, weight, yin = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        B = ctx.B
        num_res = ctx.num_res
        d = dy.shape[-1]
        dy = dy.reshape(-1, d) #dy.shape[-1])
        M, N = dy.shape #[1]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        w_arg = weight.reshape(-1, weight.shape[-1])
        # M, N = x_arg.shape
        dz = torch.empty_like(dy)
        dx = torch.empty_like(dy)
        dw = torch.empty_like(dy)
        db = torch.empty_like(dy)




        _drcln_bwd_dx_fused[(M,)](dy, dresidual, dz, dx, yin, w_arg, dw, db,  mean, rstd,  dropout_mask, ctx.p,
                                    dy.stride(0), N, #ctx.eps,
                                    BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                                    GROUP_SIZE_M=GROUP_SIZE_M,
                                    num_warps=ctx.num_warps)
        
    
        dz = dz.reshape(B, num_res, d)
        dx = dx.reshape(B, num_res, d)
        dw = dw.reshape(B, num_res, d)
        db = db.reshape(B, num_res, d)
        return dz, dx, None, dw, db, None, None, None


dracln = DropoutResAddCLN.apply

def vanilla_conditional_drcln(z, x, weight, bias, eps, mask=None, p=0.5): #, weight, bias, eps=1e-5):
    # vanilla CLN --> different scale (weight) and shift (bias) params per element in batch
    # B, M, N = x.size()
    if mask is None:
        z_out = F.dropout(z, p=p)
    else:
        assert mask.size() == z.size()
        z_out = z *mask/(1-p)
    yin =  x + z_out
    return vanilla_conditional_layer_norm(yin, weight, bias, eps=eps), yin, mask

def vanilla_conditional_layer_norm(x, weight, bias, eps=1e-5):
    # vanilla CLN --> different scale (weight) and shift (bias) params per element in batch
    # M, N = x.size()
    assert weight.size() == x.size()
    assert bias.size() == x.size()
    mean = torch.mean(x, -1)[..., None]
    var = torch.var(x, -1)[..., None]
    normalized_x = (x - mean) / torch.sqrt(var + eps)
    
    out = weight * normalized_x + bias
    return out


# @pytest.fixture
def test_drcln(B, M, N, dtype, eps=1e-5, device='cuda'):
    
    # create data
    x_shape = (B, M, N)
    # for conditional layer norm, weights and biases are (M, N) not (N, ) -- different wts/biases per row
    weight = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    z = 7+ 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    z.requires_grad_(True)

    p = 0.5
    # forward pass
    seed = torch.randint(low=0, high=65536, size=(1,))[0].item()
    y_out_tri, y_in_tri, mask_out_tri = dracln(z, x, p, weight, bias, eps, seed) 
    y_out_ref, y_in_ref, _ = vanilla_conditional_drcln(z, x, weight, bias, eps, mask_out_tri, p=p) 


    # # # backward pass (triton)
    (y_out_tri + 2* y_in_tri).backward(dy, retain_graph=True)
    dz_tri, dx_tri, dw_tri, db_tri,  = [_.grad.clone() for _ in [z, x, weight, bias ]] #, weight, bias]]
    x.grad, z.grad, weight.grad, bias.grad = None, None, None, None
    # backward pass (torch)
    (y_out_ref + 2* y_in_ref).backward(dy, retain_graph=True)
    dz_ref, dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [z, x, weight, bias]] #, weight, bias]]
    # compare
    assert torch.allclose(y_out_tri, y_out_ref, atol=1e-2, rtol=0),   (y_out_tri, y_out_ref)
    assert torch.allclose(y_in_tri, y_in_ref, atol=1e-2, rtol=0),   (y_in_tri, y_in_ref)
    # assert torch.allclose(dy_in_tri, dy_in_ref, atol=1e-2, rtol=0),   (dy_in_tri, dy_in_ref)
    assert torch.allclose(dz_tri, dz_ref, atol=1e-2, rtol=0), (dz_tri, dz_ref)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0), (dx_tri, dx_ref)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0), (db_tri, db_ref)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0), (dw_tri, dw_ref)
    
    print("✅ Triton and Torch match for DRACLN")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='dracln-forward',
        args={'B':10, 'M': 4096, 'dtype': torch.float16, 'mode': 'forward'}
    )
)
def bench_drcln(B, M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # create data
    x_shape = (B, M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    z = 7.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    z.requires_grad_(True)
    # quantiles = [0.5, 0.2, 0.8]
    p = 0.5
    seed = torch.randint(low=0, high=65536, size=(1,))[0].item()
    # utility functions
    if provider == 'triton':
        def y_fwd(): return dracln(z, x, p, weight, bias, eps, seed) 
    if provider == 'torch':
        def y_fwd(): return vanilla_conditional_drcln(z, x, weight, bias, eps, p=p) 
    if provider == 'apex':
        apex_drcln = apex.normalization.FusedDropoutResAddCLN(
            w_shape).to(x.device).to(x.dtype)
        def y_fwd(): return apex_drcln(x)  
    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms = triton.testing.do_bench_cudagraph(y_fwd)
    # backward pass
    if mode == 'backward':
        def gbps(ms): return 3 * x.numel() * x.element_size() / ms * 1e-6 
        y, _ = y_fwd()
        ms = triton.testing.do_bench_cudagraph(lambda: y.backward(dy, retain_graph=True)) 
    return gbps(ms) #, gbps(max_ms), gbps(min_ms)





if __name__ == '__main__':
    test_drcln(10, 1151, 8195, torch.float16)
    bench_drcln.run(save_path='.', print_data=True)