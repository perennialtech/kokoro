import torch
import triton
import triton.language as tl
from torch import Tensor, nn
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config(
            dict(TILE_N=TILE_N, TILE_K=TILE_K),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for TILE_N in [8, 16, 32]
        for TILE_K in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [3, 4, 5]
    ],
    key=["hidden_dim"],
)
@triton.jit
def _lstm_recurrent_from_gates_kernel(
    Gates_ptr,  # (max_seq_len, hidden_dim * 8)
    C_ptr,  # (hidden_dim * 2), fp32
    Y_ptr,  # (max_seq_len, hidden_dim * 2), fp32
    Whh_ptr,  # (hidden_dim * 8, hidden_dim)
    Bias_ptr,  # (hidden_dim * 8), fp32
    base_ptr,  # int32 scalar
    length_ptr,  # int32 scalar
    local_step,  # runtime scalar, graph node parameter
    hidden_dim: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    direction = tl.program_id(1)

    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask_n = offs_n < hidden_dim

    base = tl.load(base_ptr)
    L = tl.load(length_ptr)
    logical_t = base + local_step

    # direction 0: forward
    # direction 1: reverse
    if direction == 0:
        seq_t = logical_t
        gate_dir_offset = 0
        out_dir_offset = 0
        c_dir_offset = 0
        prev_seq_t = seq_t - 1
    else:
        seq_t = L - 1 - logical_t
        gate_dir_offset = 4 * hidden_dim
        out_dir_offset = hidden_dim
        c_dir_offset = hidden_dim
        prev_seq_t = seq_t + 1

    gates_row = Gates_ptr + seq_t * (8 * hidden_dim) + gate_dir_offset
    bias_base = direction * 4 * hidden_dim
    whh_base = direction * 4 * hidden_dim * hidden_dim

    i = tl.load(gates_row + 0 * hidden_dim + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    f = tl.load(gates_row + 1 * hidden_dim + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    g = tl.load(gates_row + 2 * hidden_dim + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    o = tl.load(gates_row + 3 * hidden_dim + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )

    i += tl.load(
        Bias_ptr + bias_base + 0 * hidden_dim + offs_n, mask=mask_n, other=0.0
    ).to(tl.float32)
    f += tl.load(
        Bias_ptr + bias_base + 1 * hidden_dim + offs_n, mask=mask_n, other=0.0
    ).to(tl.float32)
    g += tl.load(
        Bias_ptr + bias_base + 2 * hidden_dim + offs_n, mask=mask_n, other=0.0
    ).to(tl.float32)
    o += tl.load(
        Bias_ptr + bias_base + 3 * hidden_dim + offs_n, mask=mask_n, other=0.0
    ).to(tl.float32)

    if logical_t > 0:
        offs_k = tl.arange(0, TILE_K)

        prev_h_base = Y_ptr + prev_seq_t * (2 * hidden_dim) + out_dir_offset

        for k0 in range(0, hidden_dim, TILE_K):
            k = k0 + offs_k
            mask_k = k < hidden_dim

            h_prev = tl.load(prev_h_base + k, mask=mask_k, other=0.0).to(tl.float32)

            whh_i = (
                Whh_ptr
                + whh_base
                + 0 * hidden_dim * hidden_dim
                + offs_n[:, None] * hidden_dim
                + k[None, :]
            )
            whh_f = (
                Whh_ptr
                + whh_base
                + 1 * hidden_dim * hidden_dim
                + offs_n[:, None] * hidden_dim
                + k[None, :]
            )
            whh_g = (
                Whh_ptr
                + whh_base
                + 2 * hidden_dim * hidden_dim
                + offs_n[:, None] * hidden_dim
                + k[None, :]
            )
            whh_o = (
                Whh_ptr
                + whh_base
                + 3 * hidden_dim * hidden_dim
                + offs_n[:, None] * hidden_dim
                + k[None, :]
            )

            mask_nk = mask_n[:, None] & mask_k[None, :]

            wi = tl.load(whh_i, mask=mask_nk, other=0.0).to(tl.float32)
            wf = tl.load(whh_f, mask=mask_nk, other=0.0).to(tl.float32)
            wg = tl.load(whh_g, mask=mask_nk, other=0.0).to(tl.float32)
            wo = tl.load(whh_o, mask=mask_nk, other=0.0).to(tl.float32)

            i += tl.sum(wi * h_prev[None, :], axis=1)
            f += tl.sum(wf * h_prev[None, :], axis=1)
            g += tl.sum(wg * h_prev[None, :], axis=1)
            o += tl.sum(wo * h_prev[None, :], axis=1)

    if logical_t == 0:
        c_prev = tl.zeros((TILE_N,), dtype=tl.float32)
    else:
        c_prev = tl.load(C_ptr + c_dir_offset + offs_n, mask=mask_n, other=0.0).to(
            tl.float32
        )

    c_new = tl.sigmoid(f) * c_prev + tl.sigmoid(i) * libdevice.tanh(g)
    h_new = tl.sigmoid(o) * libdevice.tanh(c_new)

    tl.store(C_ptr + c_dir_offset + offs_n, c_new, mask=mask_n)
    tl.store(
        Y_ptr + seq_t * (2 * hidden_dim) + out_dir_offset + offs_n, h_new, mask=mask_n
    )


@triton.jit
def _advance_base_kernel(base_ptr, amount: tl.constexpr):
    base = tl.load(base_ptr)
    tl.store(base_ptr, base + amount)


@triton.jit
def _reset_base_length_kernel(base_ptr, length_ptr, L):
    tl.store(base_ptr, 0)
    tl.store(length_ptr, L)


class FastGraphBiLSTM(nn.Module):
    """
    Inference-only bidirectional single-layer LSTM for B == 1.

    Main design:
    - precompute all input gates with one GEMM
    - run recurrent hidden-to-hidden work with graph-captured Triton chunks
    - use dynamic base + local graph step instead of racing on time_ptr
    """

    def __init__(
        self,
        lstm: nn.LSTM,
        max_seq_len: int,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if lstm.num_layers != 1:
            raise ValueError("FastGraphBiLSTM only supports num_layers == 1")
        if not lstm.bidirectional:
            raise ValueError("FastGraphBiLSTM only supports bidirectional LSTMs")
        if not lstm.bias:
            raise ValueError("FastGraphBiLSTM requires bias=True")
        if getattr(lstm, "proj_size", 0) != 0:
            raise ValueError("FastGraphBiLSTM does not support projection LSTMs")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.batch_first = lstm.batch_first
        self.max_seq_len = int(max_seq_len)

        self.input_dim = lstm.weight_ih_l0.shape[1]
        self.hidden_dim = lstm.weight_hh_l0.shape[1]

        # NOTE: PyTorch conventionally loads state/modules on CPU first, before invoking `.to("cuda")`.
        # Do not check device.type != "cuda" at this point.
        device = lstm.weight_ih_l0.device

        if compute_dtype is None:
            compute_dtype = lstm.weight_ih_l0.dtype

        # Input projection weight:
        #   original PyTorch: W_ih_fwd, W_ih_rev are (4H, I)
        #   packed GEMM form: W_ih_t is (I, 8H)
        w_ih_cat = (
            torch.cat(
                [
                    lstm.weight_ih_l0.detach(),
                    lstm.weight_ih_l0_reverse.detach(),
                ],
                dim=0,
            )
            .to(device=device, dtype=compute_dtype)
            .contiguous()
        )

        w_ih_t = w_ih_cat.t().contiguous()

        # Recurrent projection weight:
        #   packed as (8H, H)
        w_hh_cat = (
            torch.cat(
                [
                    lstm.weight_hh_l0.detach(),
                    lstm.weight_hh_l0_reverse.detach(),
                ],
                dim=0,
            )
            .to(device=device, dtype=compute_dtype)
            .contiguous()
        )

        # Combined PyTorch biases:
        #   bias = bias_ih + bias_hh
        bias_fwd = (lstm.bias_ih_l0.detach() + lstm.bias_hh_l0.detach()).to(
            device=device,
            dtype=torch.float32,
        )
        bias_rev = (
            lstm.bias_ih_l0_reverse.detach() + lstm.bias_hh_l0_reverse.detach()
        ).to(
            device=device,
            dtype=torch.float32,
        )
        bias_cat = torch.cat([bias_fwd, bias_rev], dim=0).contiguous()

        self.register_buffer("w_ih_t", w_ih_t, persistent=True)
        self.register_buffer("w_hh_cat", w_hh_cat, persistent=True)
        self.register_buffer("bias_cat", bias_cat, persistent=True)

        self._runtime_allocated = False
        self.graphs: list[tuple[int, torch.cuda.CUDAGraph]] | None = None

        graph_lengths = []
        p = 1
        while p <= self.max_seq_len:
            graph_lengths.append(p)
            p *= 2
        self.graph_lengths = tuple(reversed(graph_lengths))

    def _ensure_runtime_buffers(self):
        if self._runtime_allocated:
            return

        device = self.w_hh_cat.device
        H = self.hidden_dim
        I = self.input_dim
        max_seq_len = self.max_seq_len

        self.register_buffer(
            "_gates_x",
            torch.empty(
                (max_seq_len, 8 * H),
                device=device,
                dtype=self.w_ih_t.dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_x2d_contiguous",
            torch.empty(
                (max_seq_len, I),
                device=device,
                dtype=self.w_ih_t.dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_c",
            torch.empty(
                (2 * H,),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_out",
            torch.empty(
                (max_seq_len, 2 * H),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_base",
            torch.empty((1,), device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_length",
            torch.empty((1,), device=device, dtype=torch.int32),
            persistent=False,
        )

        self._runtime_allocated = True

    def _grid(self):
        H = self.hidden_dim
        return lambda meta: (triton.cdiv(H, meta["TILE_N"]), 2)

    def _emit_recurrent_chunk(self, n: int):
        grid = self._grid()
        H = self.hidden_dim

        for local_step in range(n):
            _lstm_recurrent_from_gates_kernel[grid](
                self._gates_x,
                self._c,
                self._out,
                self.w_hh_cat,
                self.bias_cat,
                self._base,
                self._length,
                local_step,
                H,
            )

        _advance_base_kernel[(1,)](self._base, n)

    @torch.inference_mode()
    def _capture_graphs(self):
        self._ensure_runtime_buffers()

        H = self.hidden_dim
        grid = self._grid()

        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()

        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            self._gates_x.zero_()
            self._out.zero_()

            # Warm recurrent path with logical_t > 0.
            # This forces autotuning to benchmark the real recurrent dot-product path,
            # not the cheap time-zero path.
            self._base.fill_(1)
            self._length.fill_(self.max_seq_len)

            _lstm_recurrent_from_gates_kernel[grid](
                self._gates_x,
                self._c,
                self._out,
                self.w_hh_cat,
                self.bias_cat,
                self._base,
                self._length,
                0,
                H,
            )

            _reset_base_length_kernel[(1,)](
                self._base,
                self._length,
                self.max_seq_len,
            )

            for n in self.graph_lengths:
                _advance_base_kernel[(1,)](self._base, n)

        current_stream.wait_stream(warmup_stream)

        graphs: list[tuple[int, torch.cuda.CUDAGraph]] = []

        for n in self.graph_lengths:
            self._base.zero_()
            self._length.fill_(self.max_seq_len)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._emit_recurrent_chunk(n)

            graphs.append((n, g))

        self.graphs = graphs

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        has_fastgraph_keys = any(k.startswith(prefix + "w_ih_t") for k in state_dict)
        if not has_fastgraph_keys:
            w_ih_fwd_key = prefix + "weight_ih_l0"
            if w_ih_fwd_key in state_dict:
                keys = [
                    w_ih_fwd_key,
                    prefix + "weight_ih_l0_reverse",
                    prefix + "weight_hh_l0",
                    prefix + "weight_hh_l0_reverse",
                    prefix + "bias_ih_l0",
                    prefix + "bias_ih_l0_reverse",
                    prefix + "bias_hh_l0",
                    prefix + "bias_hh_l0_reverse",
                ]
                if all(k in state_dict for k in keys):
                    compute_dtype = self.w_ih_t.dtype

                    w_ih_fwd = state_dict.pop(keys[0])
                    w_ih_rev = state_dict.pop(keys[1])
                    w_hh_fwd = state_dict.pop(keys[2])
                    w_hh_rev = state_dict.pop(keys[3])
                    b_ih_fwd = state_dict.pop(keys[4])
                    b_ih_rev = state_dict.pop(keys[5])
                    b_hh_fwd = state_dict.pop(keys[6])
                    b_hh_rev = state_dict.pop(keys[7])

                    w_ih_cat = (
                        torch.cat([w_ih_fwd, w_ih_rev], dim=0)
                        .to(dtype=compute_dtype)
                        .contiguous()
                    )
                    w_ih_t = w_ih_cat.t().contiguous()

                    w_hh_cat = (
                        torch.cat([w_hh_fwd, w_hh_rev], dim=0)
                        .to(dtype=compute_dtype)
                        .contiguous()
                    )

                    bias_fwd = (b_ih_fwd + b_hh_fwd).to(dtype=torch.float32)
                    bias_rev = (b_ih_rev + b_hh_rev).to(dtype=torch.float32)
                    bias_cat = torch.cat([bias_fwd, bias_rev], dim=0).contiguous()

                    state_dict[prefix + "w_ih_t"] = w_ih_t
                    state_dict[prefix + "w_hh_cat"] = w_hh_cat
                    state_dict[prefix + "bias_cat"] = bias_cat

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.inference_mode()
    def forward(self, x: Tensor):
        if not x.is_cuda:
            raise ValueError("input must be CUDA")
        if x.device != self.w_ih_t.device:
            raise ValueError(
                "input device must match FastGraphBiLSTM weights: "
                f"got {x.device}, expected {self.w_ih_t.device}"
            )

        if self.batch_first:
            if x.ndim != 3:
                raise ValueError("expected input shape (B, L, I)")
            B, L, I = x.shape
            if B != 1:
                raise ValueError("FastGraphBiLSTM only supports B == 1")
            x2d = x[0]
        else:
            if x.ndim != 3:
                raise ValueError("expected input shape (L, B, I)")
            L, B, I = x.shape
            if B != 1:
                raise ValueError("FastGraphBiLSTM only supports B == 1")
            x2d = x[:, 0, :]

        if L <= 0:
            raise ValueError("empty sequences are not supported")
        if L > self.max_seq_len:
            raise ValueError(
                f"sequence length {L} exceeds max_seq_len {self.max_seq_len}"
            )
        if I != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {I}, expected {self.input_dim}")
        if x2d.dtype != self.w_ih_t.dtype:
            raise TypeError(
                f"input dtype must match compute dtype {self.w_ih_t.dtype}, got {x2d.dtype}"
            )

        if self.graphs is None:
            self._capture_graphs()

        # Callers commonly form the LSTM input by transposing channel-first
        # tensors, which produces a valid but non-contiguous [L, I] view.  Stage
        # those views into a persistent workspace so the public contract matches
        # nn.LSTM without introducing per-request CUDA allocations.
        if not x2d.is_contiguous():
            x2d_staged = self._x2d_contiguous[:L]
            x2d_staged.copy_(x2d)
            x2d = x2d_staged

        # Precompute all input-side gates in one GEMM:
        #   (L, I) @ (I, 8H) -> (L, 8H)
        torch.mm(x2d, self.w_ih_t, out=self._gates_x[:L])

        # Reset only graph time state.
        # C does not need to be zeroed because logical_t == 0 uses implicit c_prev = 0.
        _reset_base_length_kernel[(1,)](self._base, self._length, L)

        remaining = L
        for n, g in self.graphs:
            while remaining >= n:
                g.replay()
                remaining -= n

        if remaining != 0:
            raise RuntimeError("internal graph decomposition failed")

        out2d = self._out[:L]

        if out2d.dtype != x.dtype:
            out2d = out2d.to(x.dtype)

        if self.batch_first:
            out = out2d.unsqueeze(0)
        else:
            out = out2d.unsqueeze(1)

        return out, None
