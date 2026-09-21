"""CPU-only algorithm tests. These are not evidence of GPU/model performance."""
import pytest
import torch

from baselines.optimizers import (
    PerturbationGenerator, apply_noise, differentiable_meta_loss, noise_stream,
    parameter_layout, two_point_step,
)


def test_two_point_matches_directional_gradient_and_uses_two_queries():
    p = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64))
    params = [("weight", p)]
    before = p.detach().clone()
    z = next(noise_stream(params, 42))[2]
    calls = []

    def objective():
        calls.append(p.detach().clone())
        return p.square().sum().item()

    positive, negative, derivative = two_point_step(params, 42, 1e-3, 1e-2, objective)
    torch.testing.assert_close(calls[0], before + 1e-3 * z)
    torch.testing.assert_close(calls[1], before - 1e-3 * z)
    assert len(calls) == 2
    assert derivative == pytest.approx(2 * torch.dot(before, z).item())
    torch.testing.assert_close(p, before - 1e-2 * derivative * z)
    assert positive != negative


def test_uniform_generator_scales_reduce_to_mezo():
    a = torch.nn.Parameter(torch.tensor([0.2, 0.3], dtype=torch.float64))
    b = torch.nn.Parameter(a.detach().clone())
    result_a = two_point_step([("w", a)], 4, 1e-3, 1e-2, lambda: a.square().sum().item())
    result_b = two_point_step([("w", b)], 4, 1e-3, 1e-2, lambda: b.square().sum().item(),
                             {"w": torch.tensor(1.0)})
    torch.testing.assert_close(a, b)
    assert result_a == result_b


def test_noise_is_regenerated_and_does_not_advance_global_rng():
    params = [("a", torch.nn.Parameter(torch.ones(2))), ("b", torch.nn.Parameter(torch.ones(3)))]
    state = torch.random.get_rng_state()
    first = [z for _, _, z in noise_stream(params, 9)]
    second = [z for _, _, z in noise_stream(params, 9)]
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    assert torch.equal(state, torch.random.get_rng_state())
    apply_noise(params, 9, 0.001)
    apply_noise(params, 9, -0.002)
    apply_noise(params, 9, 0.001)
    for _, p in params:
        torch.testing.assert_close(p, torch.ones_like(p))


def test_generator_features_normalization_and_state():
    params = [("model.embed_tokens.weight", torch.nn.Parameter(torch.arange(12.0).reshape(4, 3))),
              ("norm.weight", torch.nn.Parameter(torch.ones(3)))]
    gen = PerturbationGenerator(params, logical_batch_size=4)
    assert gen.weights == [180, 3]
    features = []
    handles = [net.register_forward_pre_hook(lambda net, args: features.append(args[0].detach().clone()))
               for net in gen.networks]
    gen.history = (2.0, 3.0)
    scales = gen.scales(params)
    expected = torch.tensor([5.5, params[0][1].var().item(), 2, 3, 1])
    torch.testing.assert_close(features[0], expected)
    weighted = sum(w * scales[name].item() ** 2 for w, (name, _) in zip(gen.weights, params))
    assert weighted / sum(gen.weights) == pytest.approx(1.0, rel=1e-5)
    saved = gen.dynamics()
    gen.reset_history()
    assert gen.history == (0.0, 0.0)
    gen.restore_dynamics(saved)
    assert gen.history == (2.0, 3.0)
    with pytest.raises(ValueError, match="layout"):
        gen.restore_dynamics(dict(previous={}, history=[0, 0]))
    for handle in handles:
        handle.remove()


def test_meta_loss_backpropagates_only_to_generator_on_tied_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(3)
    cfg = Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                     num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                     tie_word_embeddings=True, use_cache=False)
    model = Qwen2ForCausalLM(cfg).eval()
    params = list(model.named_parameters())
    assert not any(name == "lm_head.weight" for name, _ in params)
    assert len(parameter_layout(model)) == len(params)
    before = {name: p.detach().clone() for name, p in params}
    generator = PerturbationGenerator(params, 4)
    scales = generator.scales(params, detach_normalization=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    loss = differentiable_meta_loss(model, params, 7, scales, 0.5, 0.1,
                                   dict(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids)))
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in generator.parameters() if p.grad is not None) > 0
    for name, p in params:
        assert p.grad is None
        assert torch.equal(p, before[name])
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight


@pytest.mark.parametrize("epsilon,lr", [(0, 1), (1, 0), (-1, 1)])
def test_invalid_step_rejected(epsilon, lr):
    with pytest.raises(ValueError):
        two_point_step([], 1, epsilon, lr, lambda: 0)


def test_nonfinite_objective_rejected():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(FloatingPointError):
        two_point_step([("w", p)], 1, 0.01, 0.01, lambda: float("nan"))
