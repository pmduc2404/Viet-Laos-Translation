import torch
from torch import nn
import copy

class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/pdf/1910.07467.pdf.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale
    
class RotaryPositionalEmbedding(nn.Module):
    """
    This class implements Rotary Positional Embedding (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450

    In this implementation we cache the embedding for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim,
        max_seq_len=4096,
        base=10_000,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer(
            "theta",
            theta.to("cuda" if torch.cuda.is_available() else "cpu"),
            persistent=False,
        )
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len = 4096):
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x, input_pos=None):
        """
        Args:
            x (Tensor): input tensor with shape
                [bsz, seq_len, num_heads, head_dim]
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, n_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not. When
        # input_pos is provided, we're in inference mode
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, n_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [1, s, 1, n_d // 2, 2]
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, n_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, n_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
    
class KVCache(nn.Module):
    """
    Standalone nn.Module containing a kv-cache to cache past key and values
    during inference.

    Args:
        max_batch_size (int): maximum batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of
            num_kv_heads because the cache is created after we've expanded the
            key and value tensors to have the same shape as the query tensor.
            See CausalAttention for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        max_batch_size,
        max_seq_len,
        num_heads,
        head_dim,
        dtype=torch.float32,
    ):
        super().__init__()
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            persistent=False,
        )
        self.max_batch_size = max_batch_size

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
    
class CausalSelfAttention(nn.Module):
    """Multi-headed grouped query self-attention (GQA) layer introduced
    in https://arxiv.org/pdf/2305.13245v1.pdf.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    https://github.com/Lightning-AI/lit-gpt/blob/main/lit_gpt/config.py).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        head_dim (int): dimension of each head, calculated by ``embed_dim`` // ``num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (nn.Module): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value.
            If not specified, then no caching is used.
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0.

    Raises:
        ValueError: If `num_heads` % `num_kv_heads` != 0
        ValueError: If `embed_dim` % `num_heads` != 0
        ValueError: If `attn_dropout` < 0 or > 1
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        q_proj,
        k_proj,
        v_proj,
        output_proj,
        pos_embeddings,
        kv_cache=None,
        max_seq_len=4096,
        attn_dropout=0.0,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings

    def forward(self, x, y=None, mask=None, input_pos=None):
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            y (Optional[Tensor]): input tensor with shape
                [batch_size x 1 x embed_dim]
            mask (Optional[Tensor]): Optional tensor which contains the mask.
                Only used during inference. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Returns:
            Tensor: output tensor with attention applied

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim

        TODO:
            - Return the attention weights
            - Make application of positional embeddings optional
        """
        # input has shape [b, s, d]
        bsz, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        if y is None:
            y = x

        kv_seq_len = y.shape[1]

        # q has shape [b, s, num_heads * head_dim]
        # k has shape [b, s or 1, num_kv_heads * head_dim]
        # v has shape [b, s or 1, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads

        # q: [b, s, n_kv, q_per_kv, h_d]
        # k: [b, s or 1, n_kv, 1, h_d]
        # v: [b, s or 1, n_kv, 1, h_d]
        q = q.view(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
        k = k.view(bsz, kv_seq_len, self.num_kv_heads, 1, self.head_dim)
        v = v.view(bsz, kv_seq_len, self.num_kv_heads, 1, self.head_dim)

        # if needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, kv_seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, kv_seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # llama2 applies the RoPE embeddings on tensors with shape
        # [b, s or 1, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, kv_seq_len, -1, self.head_dim)
        v = v.reshape(bsz, kv_seq_len, -1, self.head_dim)

        # Apply positional embeddings
        q = self.pos_embeddings(q, input_pos)
        k = self.pos_embeddings(k, input_pos)

        # [b, n_h, s or 1, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update key-value cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(output)

def _get_clones(module, n):
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

class TransformerEncoderLayer(nn.Module):
    """Transformer layer derived from the Llama2 model. Normalization is applied
       before the attention **and** FF layer.

    Args:
        attn (CausalSelfAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm_x (nn.Module): Normalization to be applied before self-attention.
        mlp_norm (nn.Module): Normalization to be applied before the feed-forward layer.
    """

    def __init__(self, attn, mlp, sa_norm, mlp_norm):
        super().__init__()
        self.sa_norm = sa_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def forward(self, x, mask=None, input_pos=None):
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x 1 x embed_dim]
            mask (Optional[Tensor]): Optional tensor which contains the mask.
                Only used during inference. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x 1 x embed_dim]

        Notation used for tensor shapes:
            - b: batch size
            - d: embed dim

        TODO:
            - Make position of norm configurable
        """
        # Input tensor and attention output have the same shape
        # [b, 1, d]
        # Norm applied before self-attention
        attn_out = self.attn(self.sa_norm(x), mask=mask, input_pos=input_pos)

        # Residual connection; shape: [b, 1, d]
        h = attn_out + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [b, 1, d]
        out = h + mlp_out
        return out
    
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for Machine Translation tasks, based on the Transformer architecture.

    Args:
        tok_embedding (nn.Embedding): Embedding layer to convert tokens to embedding vectors.
        layer (TransformerEncoderLayer): Transformer Encoder layer, typically consisting of self-attention and feed-forward layers.
        num_layers (int): Number of encoder layers.
        max_seq_len (int): Maximum sequence length the model will handle.
        num_heads (int): Number of query heads in Multi-Head Attention.
        head_dim (int): Dimensionality of each head in self-attention.
        norm (nn.Module): Normalization layer applied to the encoder output.

    Note:
        Argument values are checked for correctness in the module where they are used.
    """

    def __init__(
        self,
        tok_embedding,
        layer,
        num_layers,
        max_seq_len,
        num_heads,
        head_dim,
        norm,
    ):
        super().__init__()

        self.tok_embedding = tok_embedding  # Token embedding layer
        self.layers = _get_clones(layer, num_layers)  # Clone the encoder layers
        self.norm = norm  # Normalization layer applied to output
        self.max_seq_len = max_seq_len  # Maximum sequence length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None  # Causal mask is not needed for encoder

    def setup_caches(self, max_batch_size, dtype=torch.float32):
        """
        Set up memory (cache) for inference, used only during inference.
        """
        for layer in self.layers:
            layer.attn.kv_cache = KVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )

    def clear_caches(self):
        """
        Clear the cache after inference.
        """
        for layer in self.layers:
            layer.attn.kv_cache = None
        self.causal_mask = None

    def forward(self, tokens, input_pos=None):
        """
        Args:
            tokens (Tensor): Input tensor with shape [b x s], b: batch size, s: sequence length.
            input_pos (Optional[Tensor]): (Used only during inference) Contains the position of the current token.

        Returns:
            Tensor: Output tensor with shape [b x s x d], b: batch size, s: sequence length, d: embedding dimension.

        Raises:
            ValueError: if `input_pos` is None when using causal_mask (only used in inference).
        """
        # Convert tokens to embeddings
        # bsz = tokens.shape[0]  # Batch size
        h = self.tok_embedding(tokens)  # Shape: [b, s, d]

        mask = None  # No need for a mask in training
        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError("Caches are setup, but the position of input token is missing")
            mask = self.causal_mask[None, None, input_pos]  # Create causal mask for inference

        # Pass through each encoder layer
        for layer in self.layers:
            h = layer(h, mask)  # [b, s, d] after each encoder layer

        # Apply normalization
        return self.norm(h).float()  # [b, s, d] after normalization

        # return h
    
class TransformerDecoderLayer(nn.Module):
    """Transformer layer derived from the Llama2 model. Normalization is applied
       before the attention **and** FF layer.

    Args:
        attn1 (CausalSelfAttention): Attention module.
        attn2 (CausalSelfAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm_x1 (nn.Module): Normalization to be applied before self-attention.
        sa_norm_x2 (nn.Module): Normalization to be applied before self-attention.
        mlp_norm (nn.Module): Normalization to be applied before the feed-forward layer.
    """

    def __init__(self, attn1, attn2, mlp, sa_norm_x1, sa_norm_x2, mlp_norm):
        super().__init__()
        self.sa_norm_x1 = sa_norm_x1
        self.sa_norm_x2 = sa_norm_x2
        self.attn1 = attn1
        self.attn2 = attn2
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def forward(self, x, y, mask=None, input_pos=None):
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            y (Tensor): input tensor with shape
                [batch_size x 1 x embed_dim]
            mask (Optional[Tensor]): Optional tensor which contains the mask.
                Only used during inference. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - d: embed dim

        TODO:
            - Make position of norm configurable
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn1(self.sa_norm_x1(x), mask=mask, input_pos=input_pos)

        # Residual connection; shape: [b, s, d]
        h = attn_out + x

        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn2(
            self.sa_norm_x2(h),
            y,
            mask=mask,
            input_pos=input_pos,
        )

        # Residual connection; shape: [b, s, d]
        h = attn_out + h

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [b, s, d]
        out = h + mlp_out
        return out
    
class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embedding (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layer (TransformerDecoderLayer): Transformer Decoder layer.
        num_layers (int): Number of Transformer Decoder layers.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by KVCache
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            KVCache
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the KVCache
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (nn.Linear): Callable that applies a linear transformation to the output of
            the decoder.

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        tok_embedding,
        layer,
        num_layers,
        max_seq_len,
        num_heads,
        head_dim,
        norm,
        output,
    ):
        super().__init__()

        self.tok_embedding = tok_embedding
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.output = output
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None

    def setup_caches(self, max_batch_size, dtype=torch.float32):
        # inference only
        for layer in self.layers:
            layer.attn1.kv_cache = KVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            layer.attn2.kv_cache = KVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )

        # causal_mask is used during inference to ensure we're attending
        # to the right tokens
        self.causal_mask = torch.tril(
            torch.ones(
                self.max_seq_len,
                self.max_seq_len,
                dtype=torch.bool,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    def clear_caches(self):
        for layer in self.layers:
            layer.attn1.kv_cache = None
            layer.attn2.kv_cache = None
        self.causal_mask = None

    def forward(self, tokens, enc, input_pos=None):
        """
        Args:
            tokens (Tensor): input tensor with shape [b x s]
            enc (Tensor): extracted feature maps encoder [b x d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape [b x s x v]

        Raises:
            ValueError: if causal_mask is set but input_pos is None

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        # bsz, seq_len = tokens.shape
        # bsz = tokens.shape[0]

        # shape: [b, s, d]
        h = self.tok_embedding(tokens)

        # enc = enc.view(bsz, 1, -1)

        mask = None
        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, None, input_pos]

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, enc, mask, input_pos)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = self.output(h).float()
        return output
    
class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed
            through activation and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (nn.Module): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        gate_proj,
        down_proj,
        up_proj,
        activation=nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)) * self.w3(x))