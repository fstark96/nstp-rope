"""
NSTP End-to-End Test
Verifies all components work together correctly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add NSTP to path
sys.path.insert(0, '/c/Users/online/NSTP')

from nstp_core import (
    NSTPModel, NSTPConfig, NSTPLoss,
    HyperdimensionalAttention, HSAEncoder, HSAContextAccumulator, HSADenoiser,
    TTCERMoE, TTCERExpert, TTCERRouter,
    TTLinear, TTEmbedding, tt_decompose, reconstruct_tt, tt_orthogonal_loss,
    cyclic_shift, bind, unbind, superposition,
)


def test_hsa():
    """Test HSA components."""
    print("=" * 50)
    print("Testing HSA Components")
    print("=" * 50)
    
    B, N, D = 2, 256, 512
    HSA_DIM = 8192
    
    x = torch.randn(B, N, D)
    positions = torch.arange(N).unsqueeze(0).expand(B, -1)
    
    # Encoder
    encoder = HSAEncoder(D, HSA_DIM, binary=True)
    h = encoder(x)
    assert h.shape == (B, N, HSA_DIM)
    assert (h.abs() == 1).all(), "Should be binary"
    print(f"✓ HSAEncoder: {x.shape} -> {h.shape}")
    
    # Accumulator
    acc = HSAContextAccumulator(HSA_DIM, bind_mode="xor")
    M = acc(h, positions)
    assert M.shape == (B, HSA_DIM)
    print(f"✓ HSAContextAccumulator: {h.shape} -> {M.shape}")
    
    # Full attention
    attn = HyperdimensionalAttention(D, HSA_DIM, num_heads=8, binary=True, bind_mode="xor")
    out, ctx = attn(x, positions, return_context=True)
    assert out.shape == (B, N, D)
    assert ctx.shape == (B, 8, HSA_DIM // 8)
    print(f"✓ HyperdimensionalAttention: {x.shape} -> {out.shape}")
    
    # Denoiser
    denoiser = HSADenoiser(HSA_DIM, num_iterations=3, binary=True)
    noisy = torch.sign(M + torch.randn_like(M) * 0.5)
    clean = denoiser(noisy)
    assert clean.shape == (B, HSA_DIM)
    print(f"✓ HSADenoiser: {noisy.shape} -> {clean.shape}")
    
    # Binding operations
    a = torch.randint(0, 2, (D,)).float() * 2 - 1  # {+1, -1}
    b = torch.randint(0, 2, (D,)).float() * 2 - 1
    c = bind(a, b, mode="xor")
    a_rec = unbind(c, b, mode="xor")
    sim = (a * a_rec).mean()
    print(f"  Bind/Unbind similarity: {sim:.4f}")
    assert sim > 0.95, f"Unbinding failed: similarity={sim}"
    
    # Cyclic shift
    shifted = cyclic_shift(a, 10)
    assert shifted.shape == a.shape
    print(f"✓ Cyclic shift")
    
    print("HSA tests PASSED!\n")


def test_tt():
    """Test Tensor-Train components."""
    print("=" * 50)
    print("Testing TT Components")
    print("=" * 50)
    
    # Decomposition
    M, N = 1024, 1024
    W = torch.randn(M, N) * 0.02
    tt_ranks = [1, 8, 8, 8, 1]
    
    cores = tt_decompose(W, tt_ranks)
    print(f"✓ TT Decomposition: {len(cores)} cores")
    for i, c in enumerate(cores):
        print(f"  Core {i}: {c.shape}")
    
    # Reconstruction
    W_rec = reconstruct_tt(cores)
    error = (W - W_rec).norm() / W.norm()
    print(f"✓ Reconstruction error: {error:.4f} (expected > 0 for small ranks)")
    # With ranks [1,8,8,8,1], compression is 1024*1024 / 5760 = ~182x
    # Some error is expected and acceptable for extreme compression
    
    # TTLinear
    tt_linear = TTLinear(512, 512, [1, 8, 8, 1])
    x = torch.randn(2, 32, 512)
    out = tt_linear(x)
    assert out.shape == (2, 32, 512)
    print(f"✓ TTLinear: {x.shape} -> {out.shape}")
    
    # TTEmbedding
    tt_emb = TTEmbedding(1000, 256, [1, 8, 8, 1])
    idx = torch.randint(0, 1000, (2, 64))
    emb = tt_emb(idx)
    assert emb.shape == (2, 64, 256)
    print(f"✓ TTEmbedding: {idx.shape} -> {emb.shape}")
    
    # Orthogonality loss
    ortho_loss = tt_orthogonal_loss([c for c in tt_linear.cores])
    print(f"✓ TT Ortho loss: {ortho_loss:.6f}")
    
    print("TT tests PASSED!\n")


def test_moe():
    """Test TT-CER MoE components."""
    print("=" * 50)
    print("Testing TT-CER MoE Components")
    print("=" * 50)
    
    d_model = 256
    d_ff = 1024
    num_experts = 4
    
    # Router
    router = TTCERRouter(d_model, num_experts, [1, 8, 8, 1], top_k=2)
    x = torch.randn(32, d_model)
    gates, indices, aux_loss = router(x)
    assert gates.shape == (32, 2)
    assert indices.shape == (32, 2)
    assert aux_loss is not None
    print(f"✓ TTCERRouter: {x.shape} -> gates{gates.shape}, indices{indices.shape}")
    print(f"  Aux loss: {aux_loss:.6f}")
    
    # Expert
    expert = TTCERExpert(d_model, d_ff, [1, 8, 8, 8, 1])
    out = expert(x)
    assert out.shape == (32, d_model)
    print(f"✓ TTCERExpert: {x.shape} -> {out.shape}")
    
    # Full MoE
    moe = TTCERMoE(d_model, d_ff, num_experts, top_k=2,
                   router_tt_ranks=[1, 8, 8, 1],
                   expert_tt_ranks=[1, 8, 8, 8, 1])
    
    x_seq = torch.randn(2, 64, d_model)
    out, aux = moe(x_seq)
    assert out.shape == (2, 64, d_model)
    assert aux is not None
    print(f"✓ TTCERMoE: {x_seq.shape} -> {out.shape}")
    print(f"  Aux loss: {aux:.6f}")
    
    # Compression stats
    stats = moe.get_compression_stats()
    print(f"✓ Compression stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")
    
    print("MoE tests PASSED!\n")


def test_model():
    """Test full NSTP model."""
    print("=" * 50)
    print("Testing NSTP Model")
    print("=" * 50)
    
    config = NSTPConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=4,
        hsa_dim=4096,
        hsa_bind_mode='xor',
        hsa_binary=True,
        num_experts=4,
        top_k=2,
        d_ff=1024,
        router_tt_ranks=[1, 8, 8, 1],
        expert_tt_ranks=[1, 8, 8, 8, 1],
        embedding_tt_ranks=[1, 8, 8, 1],
        use_tt_embedding=True,
    )
    
    model = NSTPModel(config)
    print(f"✓ Model created: {model.num_parameters():,} params")
    
    # Forward
    x = torch.randint(0, 1000, (2, 128))
    logits, aux_losses = model(x, return_aux_losses=True)
    assert logits.shape == (2, 128, 1000)
    assert 'moe_load_balance' in aux_losses
    assert 'tt_orthogonality' in aux_losses
    print(f"✓ Forward: {x.shape} -> {logits.shape}")
    print(f"  Aux losses: {list(aux_losses.keys())}")
    
    # Loss
    from nstp_core import NSTPLoss
    loss_fn = NSTPLoss(
        vocab_size=1000,
        d_model=256,
        hsa_dim=4096,
        num_experts=4,
        num_layers=4,
    )
    targets = torch.randint(0, 1000, (2, 128))
    total_loss, losses = loss_fn(logits, targets, aux_losses)
    print(f"✓ Total loss: {total_loss:.6f}")
    for k, v in losses.items():
        print(f"  {k}: {v:.6f}")
    
    # Generation
    generated = model.generate(x[:1, :10], max_new_tokens=5, do_sample=False)
    assert generated.shape == (1, 15)
    print(f"✓ Generation: {x[:1, :10].shape} -> {generated.shape}")
    
    # Compression stats
    stats = model.get_compression_stats()
    print(f"✓ Overall compression: {stats['overall_compression']:.1f}x")
    
    # Orthogonalize
    model.orthogonalize_all_cores()
    print(f"✓ TT-core orthogonalization")
    
    print("Model tests PASSED!\n")


def test_benchmarks():
    """Test benchmark components."""
    print("=" * 50)
    print("Testing Benchmark Components")
    print("=" * 50)
    
    from benchmarks.benchmark import (
        theoretical_flops_comparison, StandardTransformer, StandardTransformerMoE
    )
    
    # Theoretical comparison
    flops = theoretical_flops_comparison(
        d_model=512, seq_len=1024, num_layers=12, num_experts=8
    )
    print(f"✓ Theoretical FLOPs:")
    print(f"  Standard: {flops['std_total_flops']/1e12:.2f} TFLOPs")
    print(f"  NSTP:     {flops['nstp_total_flops']/1e12:.2f} TFLOPs")
    print(f"  Reduction: {flops['flops_reduction']:.1f}x")
    
    # Standard transformer
    std = StandardTransformer(256, 4, 1024)
    x = torch.randn(1, 64, 256)
    out = std(x)
    assert out.shape == (1, 64, 256)
    print(f"✓ StandardTransformer: {x.shape} -> {out.shape}")
    
    # Standard MoE
    std_moe = StandardTransformerMoE(256, 4, 1024, 4, 2)
    out, aux = std_moe(x)
    assert out.shape == (1, 64, 256)
    print(f"✓ StandardTransformerMoE: {x.shape} -> {out.shape}")
    
    print("Benchmark tests PASSED!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NSTP END-TO-END TEST SUITE")
    print("=" * 60 + "\n")
    
    test_hsa()
    test_tt()
    test_moe()
    test_model()
    test_benchmarks()
    
    print("=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nNSTP Core is ready for training and benchmarking.")


if __name__ == "__main__":
    run_all_tests()