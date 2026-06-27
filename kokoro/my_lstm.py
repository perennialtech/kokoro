import torch
import triton
import triton.language as tl
from torch import Tensor, nn
from triton.language.extra import libdevice


@triton.jit
def lstm_triton_step_kernel(
    X_ptr,  # (max_length, 1, input_dim), fp32
    C_ptr,  # (hidden_dim * 2), fp32
    Y_ptr,  # (max_length, 1, hidden_dim * 2), fp32
    weight_ih_ptr,  # (hidden_dim * 4, input_dim)
    weight_hh_ptr,  # (hidden_dim * 4, hidden_dim)
    bias_ptr,  # (hidden_dim * 4), fp32
    weight_ih_reverse_ptr,  # (hidden_dim * 4, input_dim)
    weight_hh_reverse_ptr,  # (hidden_dim * 4, hidden_dim)
    bias_reverse_ptr,  # (hidden_dim * 4), fp32
    base_time_ptr,  # scalar int32
    length_ptr,  # scalar int32
    local_step,
    input_dim: tl.constexpr,
    hidden_dim: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # One program computes one tile of hidden units for one direction.
    #
    # program_id(0): hidden tile
    # program_id(1): direction, 0 forward, 1 reverse
    #
    # Batch size is intentionally fixed to 1 by the Python module.
    pid_n = tl.program_id(0)
    is_reverse = tl.program_id(1)

    base_time = tl.load(base_time_ptr)
    length = tl.load(length_ptr)
    time = base_time + local_step

    offsets_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offsets_k = tl.arange(0, TILE_K)
    mask_n = offsets_n < hidden_dim

    if is_reverse == 0:
        seq_t = time
        direction_offset = 0
        prev_hidden_delta = -hidden_dim * 2

        wih_ptr = weight_ih_ptr
        whh_ptr = weight_hh_ptr
        b_ptr = bias_ptr
    else:
        seq_t = length - 1 - time
        direction_offset = hidden_dim
        prev_hidden_delta = hidden_dim * 2

        wih_ptr = weight_ih_reverse_ptr
        whh_ptr = weight_hh_reverse_ptr
        b_ptr = bias_reverse_ptr

    x_base = X_ptr + seq_t * input_dim
    y_base = Y_ptr + seq_t * hidden_dim * 2 + direction_offset
    h_prev_base = y_base + prev_hidden_delta
    c_base = C_ptr + direction_offset

    i_gate = tl.zeros((TILE_N,), dtype=tl.float32)
    f_gate = tl.zeros((TILE_N,), dtype=tl.float32)
    g_gate = tl.zeros((TILE_N,), dtype=tl.float32)
    o_gate = tl.zeros((TILE_N,), dtype=tl.float32)

    # Input projection:
    #
    # weight_ih is stored as:
    #   [W_ii]
    #   [W_if]
    #   [W_ig]
    #   [W_io]
    #
    # Each gate matrix has shape (hidden_dim, input_dim).
    for k0 in range(0, input_dim, TILE_K):
        k = k0 + offsets_k
        mask_k = k < input_dim

        x = tl.load(x_base + k, mask=mask_k, other=0.0).to(tl.float32)

        weight_offsets = offsets_n[:, None] * input_dim + k[None, :]
        weight_mask = mask_n[:, None] & mask_k[None, :]

        w_i = tl.load(
            wih_ptr + hidden_dim * input_dim * 0 + weight_offsets,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        w_f = tl.load(
            wih_ptr + hidden_dim * input_dim * 1 + weight_offsets,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        w_g = tl.load(
            wih_ptr + hidden_dim * input_dim * 2 + weight_offsets,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        w_o = tl.load(
            wih_ptr + hidden_dim * input_dim * 3 + weight_offsets,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)

        i_gate += tl.sum(w_i * x[None, :], axis=1)
        f_gate += tl.sum(w_f * x[None, :], axis=1)
        g_gate += tl.sum(w_g * x[None, :], axis=1)
        o_gate += tl.sum(w_o * x[None, :], axis=1)

    # Recurrent projection.
    #
    # At time == 0, the initial hidden state is defined as zero, so the
    # recurrent contribution is skipped instead of reading a sentinel buffer.
    if time > 0:
        for k0 in range(0, hidden_dim, TILE_K):
            k = k0 + offsets_k
            mask_k = k < hidden_dim

            h_prev = tl.load(
                h_prev_base + k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)

            weight_offsets = offsets_n[:, None] * hidden_dim + k[None, :]
            weight_mask = mask_n[:, None] & mask_k[None, :]

            w_i = tl.load(
                whh_ptr + hidden_dim * hidden_dim * 0 + weight_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)
            w_f = tl.load(
                whh_ptr + hidden_dim * hidden_dim * 1 + weight_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)
            w_g = tl.load(
                whh_ptr + hidden_dim * hidden_dim * 2 + weight_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)
            w_o = tl.load(
                whh_ptr + hidden_dim * hidden_dim * 3 + weight_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)

            i_gate += tl.sum(w_i * h_prev[None, :], axis=1)
            f_gate += tl.sum(w_f * h_prev[None, :], axis=1)
            g_gate += tl.sum(w_g * h_prev[None, :], axis=1)
            o_gate += tl.sum(w_o * h_prev[None, :], axis=1)

    i_gate += tl.load(
        b_ptr + hidden_dim * 0 + offsets_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    f_gate += tl.load(
        b_ptr + hidden_dim * 1 + offsets_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    g_gate += tl.load(
        b_ptr + hidden_dim * 2 + offsets_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    o_gate += tl.load(
        b_ptr + hidden_dim * 3 + offsets_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)

    c_prev = tl.load(c_base + offsets_n, mask=mask_n, other=0.0).to(tl.float32)

    c = tl.sigmoid(f_gate) * c_prev + tl.sigmoid(i_gate) * libdevice.tanh(g_gate)
    h = tl.sigmoid(o_gate) * libdevice.tanh(c)

    tl.store(c_base + offsets_n, c, mask=mask_n)
    tl.store(y_base + offsets_n, h, mask=mask_n)


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _powers_of_two_descending(max_length: int) -> list[int]:
    p = 1 << (max_length.bit_length() - 1)
    lengths: list[int] = []
    while p > 0:
        lengths.append(p)
        p >>= 1
    return lengths


def _default_tile_n(hidden_dim: int) -> int:
    return 32 if hidden_dim >= 128 else 16


def _default_tile_k(input_dim: int, hidden_dim: int) -> int:
    return 64 if max(input_dim, hidden_dim) >= 64 else 32


def lstm_triton(
    x: Tensor,
    c: Tensor,
    out: Tensor,
    weights: list[Tensor],
    base_time: Tensor,
    length: Tensor,
    num_steps: int,
    input_dim: int,
    hidden_dim: int,
    tile_n: int,
    tile_k: int,
    num_warps: int,
    num_stages: int,
) -> Tensor:
    grid = (triton.cdiv(hidden_dim, tile_n), 2)

    for local_step in range(num_steps):
        lstm_triton_step_kernel[grid](
            x,
            c,
            out,
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            weights[4],
            weights[5],
            base_time,
            length,
            local_step,
            input_dim,
            hidden_dim,
            TILE_N=tile_n,
            TILE_K=tile_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return out


class MyLSTM(nn.Module):
    """
    CUDA-graph inference replacement for a narrow but common LSTM case:

    - one layer
    - bidirectional
    - bias enabled
    - no projection
    - batch size exactly 1
    - zero initial hidden and cell state
    - CUDA inference only

    The implementation computes internally in fp32, returns output and state in
    the input dtype, and returns independent cloned tensors so later calls do not
    mutate previously returned outputs.

    Unlike the original version, this implementation does not mutate a global
    time counter from inside the Triton grid. Each captured graph chunk receives
    a dynamic base timestep through device memory, and each kernel node has a
    fixed local step. This avoids the unsafe intra-kernel race caused by trying
    to update time_ptr from one program while other programs in the same launch
    might not have loaded it yet.
    """

    def __init__(
        self,
        lstm: nn.LSTM,
        max_length: int = 512,
        *,
        tile_n: int | None = None,
        tile_k: int | None = None,
        num_warps: int = 4,
        num_stages: int = 4,
    ):
        super().__init__()

        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        if lstm.num_layers != 1:
            raise ValueError("MyLSTM only supports num_layers == 1")

        if not lstm.bidirectional:
            raise ValueError("MyLSTM only supports bidirectional LSTMs")

        if not lstm.bias:
            raise ValueError("MyLSTM requires bias=True")

        if getattr(lstm, "proj_size", 0) != 0:
            raise ValueError("MyLSTM does not support projection LSTMs")

        if tile_n is not None and not _is_power_of_two(tile_n):
            raise ValueError("tile_n must be a positive power of two")

        if tile_k is not None and not _is_power_of_two(tile_k):
            raise ValueError("tile_k must be a positive power of two")

        if num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError("num_warps must be one of 1, 2, 4, 8, 16, or 32")

        if not isinstance(num_stages, int) or num_stages <= 0:
            raise ValueError("num_stages must be a positive integer")

        self.max_length = max_length
        self.batch_first = lstm.batch_first
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.num_warps = num_warps
        self.num_stages = num_stages

        for name in (
            "weight_ih_l0",
            "weight_hh_l0",
            "bias_ih_l0",
            "bias_hh_l0",
            "weight_ih_l0_reverse",
            "weight_hh_l0_reverse",
            "bias_ih_l0_reverse",
            "bias_hh_l0_reverse",
        ):
            value = getattr(lstm, name).detach().clone().contiguous()
            self.register_buffer(name, value)

        self._validate_weight_shapes()

        self.register_buffer("_x", None, persistent=False)
        self.register_buffer("_c", None, persistent=False)
        self.register_buffer("_out", None, persistent=False)
        self.register_buffer("_bias", None, persistent=False)
        self.register_buffer("_bias_reverse", None, persistent=False)
        self.register_buffer("_base_time", None, persistent=False)
        self.register_buffer("_length", None, persistent=False)

        self._graphs: list[tuple[int, torch.cuda.CUDAGraph]] | None = None
        self._captured_signature = None

    @property
    def input_size(self) -> int:
        return int(self.weight_ih_l0.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.weight_hh_l0.shape[1])

    def _apply(self, fn):
        # Moving or dtype-converting a module invalidates any captured CUDA graph,
        # because graph nodes contain concrete device pointers.
        self.reset_cuda_graphs()
        return super()._apply(fn)

    def reset_cuda_graphs(self) -> None:
        self._graphs = None
        self._captured_signature = None

    def _validate_weight_shapes(self) -> None:
        input_dim = int(self.weight_ih_l0.shape[1])
        hidden_dim = int(self.weight_hh_l0.shape[1])

        expected_wih = (hidden_dim * 4, input_dim)
        expected_whh = (hidden_dim * 4, hidden_dim)
        expected_bias = (hidden_dim * 4,)

        tensors_and_shapes = (
            (self.weight_ih_l0, expected_wih, "weight_ih_l0"),
            (self.weight_hh_l0, expected_whh, "weight_hh_l0"),
            (self.bias_ih_l0, expected_bias, "bias_ih_l0"),
            (self.bias_hh_l0, expected_bias, "bias_hh_l0"),
            (self.weight_ih_l0_reverse, expected_wih, "weight_ih_l0_reverse"),
            (self.weight_hh_l0_reverse, expected_whh, "weight_hh_l0_reverse"),
            (self.bias_ih_l0_reverse, expected_bias, "bias_ih_l0_reverse"),
            (self.bias_hh_l0_reverse, expected_bias, "bias_hh_l0_reverse"),
        )

        for tensor, expected_shape, name in tensors_and_shapes:
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}"
                )

            if not tensor.is_floating_point():
                raise ValueError(f"{name} must be a floating point tensor")

            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    def _runtime_tiles(self) -> tuple[int, int]:
        input_dim = self.input_size
        hidden_dim = self.hidden_size

        tile_n = self.tile_n if self.tile_n is not None else _default_tile_n(hidden_dim)
        tile_k = (
            self.tile_k
            if self.tile_k is not None
            else _default_tile_k(
                input_dim,
                hidden_dim,
            )
        )

        return tile_n, tile_k

    def _graph_signature(self):
        tensors = (
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            self.weight_ih_l0_reverse,
            self.weight_hh_l0_reverse,
            self.bias_ih_l0_reverse,
            self.bias_hh_l0_reverse,
            self._x,
            self._c,
            self._out,
            self._bias,
            self._bias_reverse,
            self._base_time,
            self._length,
        )

        tensor_signature = tuple(
            (
                None
                if t is None
                else (
                    str(t.device),
                    t.dtype,
                    tuple(t.shape),
                    tuple(t.stride()),
                    t.data_ptr(),
                )
            )
            for t in tensors
        )

        return (
            self.max_length,
            self.input_size,
            self.hidden_size,
            self._runtime_tiles(),
            self.num_warps,
            self.num_stages,
            tensor_signature,
        )

    def _allocate_static_buffers(self) -> None:
        device = self.weight_ih_l0.device
        input_dim = self.input_size
        hidden_dim = self.hidden_size

        self._x = torch.empty(
            (self.max_length, 1, input_dim),
            device=device,
            dtype=torch.float32,
        )
        self._c = torch.empty(
            (hidden_dim * 2,),
            device=device,
            dtype=torch.float32,
        )
        self._out = torch.empty(
            (self.max_length, 1, hidden_dim * 2),
            device=device,
            dtype=torch.float32,
        )
        self._bias = torch.empty(
            (hidden_dim * 4,),
            device=device,
            dtype=torch.float32,
        )
        self._bias_reverse = torch.empty(
            (hidden_dim * 4,),
            device=device,
            dtype=torch.float32,
        )
        self._base_time = torch.empty(
            (1,),
            device=device,
            dtype=torch.int32,
        )
        self._length = torch.empty(
            (1,),
            device=device,
            dtype=torch.int32,
        )

        self._x.zero_()
        self._c.zero_()
        self._out.zero_()
        self._base_time.zero_()
        self._length.zero_()

    @torch.inference_mode()
    def _refresh_combined_biases(self) -> None:
        torch.add(self.bias_ih_l0, self.bias_hh_l0, out=self._bias)
        torch.add(
            self.bias_ih_l0_reverse,
            self.bias_hh_l0_reverse,
            out=self._bias_reverse,
        )

    def _weights_for_kernel(self) -> list[Tensor]:
        return [
            self.weight_ih_l0,
            self.weight_hh_l0,
            self._bias,
            self.weight_ih_l0_reverse,
            self.weight_hh_l0_reverse,
            self._bias_reverse,
        ]

    def _check_cuda_ready(self) -> None:
        self._validate_weight_shapes()

        tensors = (
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            self.weight_ih_l0_reverse,
            self.weight_hh_l0_reverse,
            self.bias_ih_l0_reverse,
            self.bias_hh_l0_reverse,
        )

        device = tensors[0].device

        if not device.type == "cuda":
            raise RuntimeError(
                "MyLSTM requires CUDA tensors. Move the module to CUDA first."
            )

        for tensor in tensors:
            if tensor.device != device:
                raise RuntimeError(
                    "all LSTM weights and biases must be on the same device"
                )

            if not tensor.is_contiguous():
                raise RuntimeError("all LSTM weights and biases must be contiguous")

    @torch.inference_mode()
    def _capture_graphs(self) -> None:
        self._check_cuda_ready()
        self._allocate_static_buffers()
        self._refresh_combined_biases()

        input_dim = self.input_size
        hidden_dim = self.hidden_size
        tile_n, tile_k = self._runtime_tiles()
        weights = self._weights_for_kernel()
        graph_lengths = _powers_of_two_descending(self.max_length)

        device = self.weight_ih_l0.device
        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)

        graphs: list[tuple[int, torch.cuda.CUDAGraph]] = []

        for graph_length in graph_lengths:
            self._length.fill_(self.max_length)

            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    self._c.zero_()
                    self._out.zero_()
                    self._base_time.zero_()

                    lstm_triton(
                        self._x,
                        self._c,
                        self._out,
                        weights,
                        self._base_time,
                        self._length,
                        graph_length,
                        input_dim,
                        hidden_dim,
                        tile_n,
                        tile_k,
                        self.num_warps,
                        self.num_stages,
                    )

            current_stream.wait_stream(warmup_stream)

            self._c.zero_()
            self._out.zero_()
            self._base_time.zero_()
            self._length.fill_(self.max_length)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                lstm_triton(
                    self._x,
                    self._c,
                    self._out,
                    weights,
                    self._base_time,
                    self._length,
                    graph_length,
                    input_dim,
                    hidden_dim,
                    tile_n,
                    tile_k,
                    self.num_warps,
                    self.num_stages,
                )

            graphs.append((graph_length, graph))

        self._graphs = graphs
        self._captured_signature = self._graph_signature()

    def _ensure_graphs(self) -> None:
        if self._graphs is None:
            self._capture_graphs()
            return

        if self._captured_signature != self._graph_signature():
            self._capture_graphs()

    @torch.inference_mode()
    def forward(
        self,
        x: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if hx is not None:
            raise ValueError("MyLSTM only supports zero initial hidden and cell state")

        self._ensure_graphs()

        if x.ndim != 3:
            raise ValueError("input must be a 3D tensor")

        input_dtype = x.dtype

        if not x.is_floating_point():
            raise ValueError("input must be a floating point tensor")

        if self.batch_first:
            x_time_major = x.transpose(0, 1)
        else:
            x_time_major = x

        length, batch_size, input_dim = x_time_major.shape

        if batch_size != 1:
            raise ValueError("MyLSTM only supports batch size 1")

        if input_dim != self.input_size:
            raise ValueError(
                f"input has input_size={input_dim}, expected {self.input_size}"
            )

        if length > self.max_length:
            raise ValueError(
                f"input length {length} exceeds max_length {self.max_length}"
            )

        if x_time_major.device != self.weight_ih_l0.device:
            raise ValueError(
                "input must be on the same CUDA device as the MyLSTM weights"
            )

        hidden_dim = self.hidden_size

        self._refresh_combined_biases()

        self._c.zero_()
        self._out.zero_()
        self._base_time.zero_()
        self._length.fill_(length)

        if length > 0:
            self._x[:length].copy_(x_time_major, non_blocking=True)

            remaining = int(length)
            processed = 0

            for graph_length, graph in self._graphs:
                if remaining >= graph_length:
                    self._base_time.fill_(processed)
                    graph.replay()

                    processed += graph_length
                    remaining -= graph_length

            if remaining != 0:
                raise RuntimeError("internal graph decomposition failed")

        out_fp32 = self._out[:length]

        if self.batch_first:
            out_fp32 = out_fp32.transpose(0, 1)

        out = out_fp32.to(dtype=input_dtype).clone(
            memory_format=torch.contiguous_format
        )

        if length == 0:
            h_n_fp32 = torch.zeros(
                (2, 1, hidden_dim),
                device=self._out.device,
                dtype=torch.float32,
            )
        else:
            h_forward = self._out[length - 1, 0, :hidden_dim]
            h_reverse = self._out[0, 0, hidden_dim:]
            h_n_fp32 = torch.stack((h_forward, h_reverse), dim=0).unsqueeze(1)

        c_forward = self._c[:hidden_dim]
        c_reverse = self._c[hidden_dim:]
        c_n_fp32 = torch.stack((c_forward, c_reverse), dim=0).unsqueeze(1)

        h_n = h_n_fp32.to(dtype=input_dtype).clone(
            memory_format=torch.contiguous_format
        )
        c_n = c_n_fp32.to(dtype=input_dtype).clone(
            memory_format=torch.contiguous_format
        )

        return out, (h_n, c_n)
