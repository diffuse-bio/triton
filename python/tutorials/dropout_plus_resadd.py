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
        y = yinhat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


    # residual add 

    # # residual add Z + X


    # # CLN Z + X



    # # Also map the program id to the row of W and B it should compute.
    # W += row * stride
    # B += row * stride
    # # Compute mean
    # mean = 0
    # _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # # print(_mean.size(), N, BLOCK_SIZE)
    # # sum up rows
    # for off in range(0, N, BLOCK_SIZE):
    #     cols = off + tl.arange(0, BLOCK_SIZE)
    #     a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    #     _mean += a
    # # per row mean
    # mean = tl.sum(_mean, axis=0) / N
    # # print(mean.size())
    # # Compute variance -- per row
    # _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # for off in range(0, N, BLOCK_SIZE):
    #     cols = off + tl.arange(0, BLOCK_SIZE)
    #     x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    #     x = tl.where(cols < N, x - mean, 0.)
    #     _var += x * x
    # var = tl.sum(_var, axis=0) / N
    # rstd = 1 / tl.sqrt(var + eps)
    # # Write mean / rstd ( per row)
    # tl.store(Mean + row, mean)
    # tl.store(Rstd + row, rstd)
    # # Normalize and apply linear transformation
    # for off in range(0, N, BLOCK_SIZE):
    #     cols = off + tl.arange(0, BLOCK_SIZE)
    #     mask = cols < N
    #     w = tl.load(W + cols, mask=mask)
    #     b = tl.load(B + cols, mask=mask)
    #     x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    #     x_hat = (x - mean) * rstd
    #     y = x_hat * w + b
    #     # Write output
    #     tl.store(Y + cols, y, mask=mask)


@triton.jit
def _drcln_bwd_dx_fused(
    DY, # pointer to output gradient
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
    dyin = (wdy - (yin_hat * c1 + c2)) * rstd
    # Write dx
    # tl.store(DX + cols, dyin, mask=mask)
    # Accumulate partial sums for dw/db
    dw = (dy * yin_hat).to(w.dtype)
    db = (dy).to(w.dtype)
    
    tl.store(DW + cols, dw, mask=mask)
    tl.store(DB + cols, db, mask=mask)

    # dx is just dyin (res add)
    tl.store(DX + cols, dyin, mask=mask)

    # compute dz (dropout then add)
    dropout_mask = tl.load(DROPOUT_MASK + cols, mask=mask, other=0.).to(tl.float32)
    dz = dyin * dropout_mask / (1 - p)
    tl.store(DZ + cols, dz, mask=mask)



    # # Z += row * stride
    
    # DY += row * stride
    
    
    # Z_OUT += row * stride

    # z_out = tl.load(Z_OUT + cols, mask=mask, other=0).to(tl.float32)
    # dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    # w = tl.load(W + cols, mask=mask).to(tl.float32)
    # # dz = dln * mask * z / (1 - p)
    # # dx = dln


    # # dL/dz = DL/dy dy/dz
    # # z_out = mask * z / (1 - p)
    # # gradient is  dz = dy * mask / (1 - p)
    # # for off in range(0, N, BLOCK_SIZE):
    # cols = tl.arange(0, BLOCK_SIZE_N)
    # mask = cols < N
    # dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    # dropout_mask = tl.load(DROPOUT_MASK + cols, mask=mask, other=0.).to(tl.float32)
    # dz = dy * dropout_mask / (1 - p)
    # tl.store(DZ + cols, dz, mask=mask)
    # # dx = dy 
    

    # dy = dropout(z) + x #dx  = 
    # dz = 


    # # do dropout on Z
    # for off in range(0, N, BLOCK_SIZE):
    #     all_elem = row * stride + (off + tl.arange(0, BLOCK_SIZE))
    #     cols = off + tl.arange(0, BLOCK_SIZE)
    #     mask = cols < N
    #     z = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
    #     random = tl.rand(seed, all_elem)
    #     x_keep = random > p
    #     # _mean += a
    #     output = tl.where(x_keep, z / (1 - p), 0.0)
    #     tl.store(MASK_out + cols, x_keep, mask=mask)
    #     tl.store(Z_out + cols, output, mask=mask)



    # X += row * stride
    # DY += row * stride
    # DZ += row * stride


    # # Also map the program id to the elements of W it should compute (per-row)
    # W += row * stride
    # # 
    # if False:
    #     # Offset locks and weights/biases gradient pointer for parallel reduction
    #     lock_id = row % GROUP_SIZE_M
    #     Lock += lock_id
    #     Count = Lock + GROUP_SIZE_M
    #     DW = DW + lock_id * N + cols
    #     DB = DB + lock_id * N + cols
    # # Load data to SRAM
    # x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    # dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    # w = tl.load(W + cols, mask=mask).to(tl.float32)
    # # dw = tl.load(DW + cols, mask=mask).to(tl.float32)
    # # db = tl.load(DB + cols, mask=mask).to(tl.float32)

    # mean = tl.load(Mean + row)
    # rstd = tl.load(Rstd + row)
    # # Compute dx
    # xhat = (x - mean) * rstd
    # wdy = w * dy
    # xhat = tl.where(mask, xhat, 0.)
    # wdy = tl.where(mask, wdy, 0.)
    # c1 = tl.sum(xhat * wdy, axis=0) / N
    # c2 = tl.sum(wdy, axis=0) / N
    # dx = (wdy - (xhat * c1 + c2)) * rstd
    # # Write dx
    # tl.store(DX + cols, dx, mask=mask)
    # # Accumulate partial sums for dw/db
    # partial_dw = (dy * xhat).to(w.dtype)
    # partial_db = (dy).to(w.dtype)
    
    # # partial_dw = dw + (dy * xhat).to(w.dtype)
    # # partial_db = db + (dy).to(w.dtype)
    # if False:
    #     while tl.atomic_cas(Lock, 0, 1) == 1:
    #         pass
    #     count = tl.load(Count)
    #     # First store doesn't accumulate
    #     if count == 0:
    #         tl.atomic_xchg(Count, 1)
    #     else:
    #         partial_dw += tl.load(DW, mask=mask)
    #         partial_db += tl.load(DB, mask=mask)
    # # no reduction needed (no sum over batch to get grads DW and DB)
    # DW += row * stride
    # DB += row * stride
    # tl.store(DW + cols, partial_dw, mask=mask)
    # tl.store(DB + cols, partial_db, mask=mask)
    # if False:
    #     # Release the lock
    #     tl.atomic_xchg(Lock, 0)


# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.


class DropoutResAddCLN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, x, p, weight, bias, eps, seed=42):
        # z --> input ahead of dropout
        # x --> original input (residual add)

        # allocate output
        y = torch.empty_like(x)
        yin = torch.empty_like(z)
        mask_out = torch.empty_like(z)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
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



        _drcln_fwd_fused[(M,)](x_arg, y, z, yin, mask_out, p, seed, weight, bias, mean, rstd,
                                    x_arg.stride(0), N, eps,
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(mask_out, mean, rstd, weight, yin ) #x, weight, bias, mean, rstd)
        # print('mean kernel', mean) #, 1/rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        # ctx.mask_out = mask_out
        ctx.p = p
        return y #z_out #, mask_out

    @staticmethod
    def backward(ctx, dy):
        dropout_mask, mean, rstd, weight, yin = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        M, N = dy.shape #[1]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        # x_arg = x.reshape(-1, x.shape[-1])
        # M, N = x_arg.shape
        dz = torch.empty_like(dy)
        dx = torch.empty_like(dy)
        dw = torch.empty_like(dy)
        db = torch.empty_like(dy)



        _drcln_bwd_dx_fused[(M,)](dy, dz, dx, yin, weight, dw, db,  mean, rstd,  dropout_mask, ctx.p,
                                    dy.stride(0), N, #ctx.eps,
                                    BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                                    GROUP_SIZE_M=GROUP_SIZE_M,
                                    num_warps=ctx.num_warps)
        
    
        
        return dz, dx, None, dw, db, None
        # # dw and db are now (M, N) instead of N
        # dw = torch.empty((w.shape[0], w.shape[1]), dtype=w.dtype, device=w.device)
        # db = torch.empty((w.shape[0], w.shape[1]), dtype=w.dtype, device=w.device)
        # if False:
        #     # allocate output
        #     locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
        #     _dw = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device)
        #     _db = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device)
        #     dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        #     db = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        # dx = torch.empty_like(dy)
        # # enqueue kernel using forward pass heuristics
        # # also compute partial sums for DW and DB
        # x_arg = x.reshape(-1, x.shape[-1])
        # M, N = x_arg.shape
        # _drcln_bwd_dx_fused[(M,)](dx, dy, dw, db, x, w, b, m, v, None,
        #                                x_arg.stride(0), N, ctx.eps,
        #                                BLOCK_SIZE_N=ctx.BLOCK_SIZE,
        #                                GROUP_SIZE_M=GROUP_SIZE_M,
        #                                num_warps=ctx.num_warps)
        # # no need to accumulate partial sums 
        # if False:
            
        #     grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        #     # accumulate partial sums in separate kernel
        #     _drcln_bwd_dwdb[grid](_dw, _db, dw, db, GROUP_SIZE_M, N,
        #                             BLOCK_SIZE_M=32,
        #                             BLOCK_SIZE_N=128, num_ctas=1)

        return dz #, None
        # return dx, None, dw, db, None


dracln = DropoutResAddCLN.apply

def vanilla_conditional_drcln(z, x, weight, bias, eps, p=0.5): #, weight, bias, eps=1e-5):
    # vanilla CLN --> different scale (weight) and shift (bias) params per element in batch
    M, N = x.size()
    p = 0.5
    z_out = F.dropout(z, p=p)
    yin =  x + z_out
    # return y
    return vanilla_conditional_layer_norm(yin, weight, bias, eps=1e-5)
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()


    assert weight.size() == x.size()
    assert bias.size() == x.size()
    mean = torch.mean(x, -1)[..., None]
    var = torch.var(x, -1)[..., None]
    normalized_x = (x - mean) / torch.sqrt(var + eps)

    out = weight * normalized_x + bias
    return out

def vanilla_conditional_layer_norm(x, weight, bias, eps=1e-5):
    # vanilla CLN --> different scale (weight) and shift (bias) params per element in batch
    M, N = x.size()
    assert weight.size() == x.size()
    assert bias.size() == x.size()
    mean = torch.mean(x, -1)[..., None]
    # print('mean', mean)
    var = torch.var(x, -1)[..., None]
    normalized_x = (x - mean) / torch.sqrt(var + eps)

    out = weight * normalized_x + bias
    return out


# @pytest.fixture
def test_drcln(M, N, dtype, eps=1e-5, device='cuda'):
    
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    # weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    # bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    # for conditional layer norm, weights and biases are (M, N) not (N, ) -- different wts/biases per row
    weight = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    z = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    z.requires_grad_(True)

    p = 0.5
    # forward pass

    z_out_tri = dracln(z, x, p, weight, bias, eps) #, w_shape) #, p=0.5, seed=123)
    print(z_out_tri[0], z_out_tri[1], z_out_tri[-1])
    # print(mask_out_tri[0], mask_out_tri[1], mask_out_tri[-1])
    # y_tri = layer_norm(x, w_shape, weight, bias, eps)
    z_out_ref = vanilla_conditional_drcln(z, x, weight, bias, eps).to(dtype) #torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    print(z_out_ref[0], z_out_ref[-1])



    # # # backward pass (triton)
    z_out_tri.backward(dy, retain_graph=True)
    # dw_tri, db_tri
    dz_tri, dx_tri, dw_tri, db_tri  = [_.grad.clone() for _ in [z, x, weight, bias, ]] #, weight, bias]]
    # print(dz_tri.size())
    print(dz_tri[0], dz_tri[1], dz_tri[-1])
    x.grad = None #, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    z_out_ref.backward(dy, retain_graph=True)
    dz_ref, dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [z, x, weight, bias,]] #, weight, bias]]
    # # # compare
    print(dz_ref[0], dz_ref[1], dz_ref[-1])
    

    # # assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    # assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0), (dw_tri, dw_ref)
    # assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0), (db_tri, db_ref)
    # assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0), (dx_tri, dx_ref)
    # print("✅ Triton and Torch for bwd pass -- residual add")
    # assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    # assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
    # print("✅ Triton and Torch match")


# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['N'],
#         x_vals=[512 * i for i in range(2, 32)],
#         line_arg='provider',
#         line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
#         line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
#         styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
#         ylabel='GB/s',
#         plot_name='layer-norm-backward',
#         args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}
#     )
# )
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='dracln-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}
    )
)
def bench_drcln(M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    z = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    z.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    p = 0.5
    # utility functions
    if provider == 'triton':
        #  z, x, p, weight, bias, eps,
        def y_fwd(): return dracln(z, x, p, weight, bias, eps) #dracln(x, w_shape, weight, bias, eps)  # noqa: F811, E704
    if provider == 'torch':
        def y_fwd(): return vanilla_conditional_drcln(z, x, weight, bias, eps, p=p)  # noqa: F811, E704
    if provider == 'apex':
        apex_drcln = apex.normalization.FusedDropoutResAddCLN(
            w_shape).to(x.device).to(x.dtype)
        def y_fwd(): return apex_drcln(x)  # noqa: F811, E704
    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms = triton.testing.do_bench_cudagraph(y_fwd)
        # ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        def gbps(ms): return 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704
        y = y_fwd()
        ms = triton.testing.do_bench_cudagraph(lambda: y.backward(dy, retain_graph=True)) 
        # ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True),
                                                    #  quantiles=quantiles, grad_to_none=[x], rep=500)
    return gbps(ms) #, gbps(max_ms), gbps(min_ms)





# x = torch.randn(size=(10, 20,)).cuda()
# # Compare this to the baseline - dropout mask is never instantiated!
# output = seeded_dropout(x, p=0.5, seed=123)
# output2 = seeded_dropout(x, p=0.5, seed=123)
# output3 = seeded_dropout(x, p=0.5, seed=512)

# print(tabulate.tabulate([
#     ["input"] + x.tolist(),
#     ["output (seed = 123)"] + output.tolist(),
#     ["output (seed = 123)"] + output2.tolist(),
#     ["output (seed = 512)"] + output3.tolist()
# ]))


test_drcln(1151, 10, torch.float16)
bench_drcln.run(save_path='.', print_data=True)
# %%
# References
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
