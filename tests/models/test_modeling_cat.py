# Tests for CAT (Compress And Attend Transformers)

import pytest
import torch

from fla.models.cat import CATConfig, CATForCausalLM
from fla.utils import device


def create_cat_model(
    L: int,
    H: int,
    D: int,
    max_chunk_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    max_position_embeddings: int | None = None,
):
    """Create a CAT model with the given configuration."""
    hidden_size = H * D
    config_kwargs = dict(
        hidden_size=hidden_size,
        num_hidden_layers=L,
        num_heads=H,
        max_chunk_size=max_chunk_size,
        dim_fx=hidden_size,  # Must equal hidden_size
        compressor_hidden_size=hidden_size // 2,
        compressor_num_layers=max(1, L // 4),
        compressor_num_heads=max(1, H // 2),
        vocab_size=1000,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
    )
    if max_position_embeddings is not None:
        config_kwargs["max_position_embeddings"] = max_position_embeddings
    config = CATConfig(**config_kwargs)
    model = CATForCausalLM(config)
    model.to(dtype).to(device)
    return model, config


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'max_chunk_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-chunk{}-{}".format(*test))
        for test in [
            # Basic tests with different configurations
            (2, 2, 64, 4, 64, 8, torch.bfloat16),
            (2, 2, 64, 4, 64, 16, torch.bfloat16),
            (4, 2, 128, 4, 64, 16, torch.bfloat16),
        ]
    ],
)
def test_cat_forward_backward(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    max_chunk_size: int,
    dtype: torch.dtype,
):
    """Test forward and backward pass of CAT model."""
    model, config = create_cat_model(L, H, D, max_chunk_size, dtype)
    
    # Forward pass with default chunk_size (max_chunk_size)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    output = model(input_ids, output_hidden_states=True)
    
    assert output.logits is not None
    assert output.logits.shape == (B, T, config.vocab_size)
    
    # Backward pass
    loss = output.logits.sum()
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'max_chunk_size', 'test_chunk_size', 'max_pos_emb', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-max{}-test{}-{}".format(test[0], test[1], test[2], test[3], test[4], test[5], test[6], test[8]))
        for test in [
            # Test adaptive chunk sizes (T and max_position_embeddings divisible by chunk sizes)
            (2, 2, 64, 4, 64, 16, 8, None, torch.bfloat16),   # 64/16=4, 64/8=8
            (2, 2, 64, 4, 64, 16, 4, None, torch.bfloat16),   # 64/16=4, 64/4=16
            (2, 2, 96, 4, 64, 16, 12, 96, torch.bfloat16),    # 96/16=6, 96/12=8 (needs max_pos=96 for chunk 12)
        ]
    ],
)
def test_cat_adaptive_chunk_size(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    max_chunk_size: int,
    test_chunk_size: int,
    max_pos_emb: int | None,
    dtype: torch.dtype,
):
    """Test CAT model with different chunk sizes."""
    model, config = create_cat_model(L, H, D, max_chunk_size, dtype, max_position_embeddings=max_pos_emb)
    model.eval()
    
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    
    with torch.no_grad():
        # Forward with any chunk size
        output = model(input_ids, chunk_size=test_chunk_size)
        
        assert output.logits is not None
        assert output.logits.shape == (B, T, config.vocab_size)


def test_cat_compiled_forward():
    """Test CAT model forward pass with full torch.compile."""
    model, config = create_cat_model(2, 4, 64, max_chunk_size=16, dtype=torch.bfloat16)
    model.eval()

    compiled_model = torch.compile(model, mode="default", dynamic=False)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 64), device=device)

    with torch.no_grad():
        # Warm-up runs to trigger compilation and stabilize
        for _ in range(3):
            _ = compiled_model(input_ids)

        output = compiled_model(input_ids)

    assert output.logits is not None
    assert output.logits.shape == (2, 64, config.vocab_size)


@pytest.mark.parametrize(
    ['chunk_size'],
    [
        pytest.param(chunk_size, id=f"chunk{chunk_size}")
        for chunk_size in [4, 8, 16]  # Test multiple chunk sizes up to max_chunk_size=16
    ],
)
def test_cat_generation(chunk_size: int):
    """Test CAT model generation via repeated forward passes."""
    model, config = create_cat_model(2, 4, 64, max_chunk_size=16, dtype=torch.bfloat16)
    model.eval()

    # Prompt of 32 tokens, generate 16 more
    prompt_len = 32
    max_new_tokens = 16
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, prompt_len), device=device)

    with torch.no_grad():
        # Generate using HF's generate (will do repeated forward passes since CAT has no KV cache)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,  # CAT doesn't use standard KV cache
            chunk_size=chunk_size,
        )

    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] == prompt_len + max_new_tokens
    # Verify prompt is preserved
    assert torch.equal(output_ids[:, :prompt_len], input_ids)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'max_chunk_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-chunk{}-{}".format(*test))
        for test in [
            (2, 2, 64, 4, 64, 16, torch.bfloat16),
        ]
    ],
)
def test_cat_with_labels(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    max_chunk_size: int,
    dtype: torch.dtype,
):
    """Test CAT model with labels for loss computation."""
    model, config = create_cat_model(L, H, D, max_chunk_size, dtype)
    
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    labels = input_ids.clone()
    
    output = model(input_ids, labels=labels)
    
    assert output.loss is not None
    assert output.loss.ndim == 0  # Scalar loss
    
    # Backward pass
    output.loss.backward()


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'max_chunk_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-chunk{}-{}".format(*test))
        for test in [
            # Test with sequence lengths not divisible by chunk_size
            (2, 2, 50, 4, 64, 16, torch.bfloat16),  # 50 % 16 != 0
            (2, 2, 100, 4, 64, 16, torch.bfloat16),  # 100 % 16 != 0
        ]
    ],
)
def test_cat_padding(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    max_chunk_size: int,
    dtype: torch.dtype,
):
    """Test CAT model with sequence lengths not divisible by chunk_size."""
    model, config = create_cat_model(L, H, D, max_chunk_size, dtype)
    model.eval()
    
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    
    with torch.no_grad():
        output = model(input_ids)
        
        # Output should be trimmed to original sequence length
        assert output.logits.shape == (B, T, config.vocab_size)


@pytest.mark.parametrize(
    ['max_chunk_size', 'test_chunk_size'],
    [
        pytest.param(*test, id="max{}-test{}".format(*test))
        for test in [
            (16, 32),  # test_chunk_size > max_chunk_size should fail
        ]
    ],
)
def test_cat_chunk_size_validation(
    max_chunk_size: int,
    test_chunk_size: int,
):
    """Test that CAT model raises error when chunk_size > max_chunk_size."""
    model, config = create_cat_model(2, 4, 64, max_chunk_size, torch.bfloat16)
    model.eval()
    
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 64), device=device)

    with pytest.raises(ValueError, match="chunk_size.*cannot exceed"):
        model(input_ids, chunk_size=test_chunk_size)
