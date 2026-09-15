"""Small independent oracle; not a trained model or a retrieval experiment."""
from __future__ import annotations

import unittest
import torch
from torch import Tensor


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and have the logits shape")
    masked = logits.masked_fill(~mask, float("-inf"))
    has_path = mask.any(dim=dim, keepdim=True)
    safe = torch.where(has_path, masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=dim) * mask.to(logits.dtype)


def validate(z: Tensor, valid: Tensor, allowed: Tensor) -> tuple[int, int, int]:
    if z.ndim != 5 or not z.is_floating_point():
        raise ValueError("z must be floating [N,N,K,K,D]")
    n, n2, k, k2, d = z.shape
    if n != n2 or k != k2 or k < 1:
        raise ValueError("inconsistent variable/candidate dimensions")
    if valid.shape != (n, k) or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [N,K]")
    if allowed.shape != (n, n, n) or allowed.dtype != torch.bool:
        raise ValueError("allowed must be bool [target_i,target_j,bridge_k]")
    if z.device != valid.device or z.device != allowed.device:
        raise ValueError("all tensors must use the same device")
    return n, k, d


def triangle_reference(z: Tensor, valid: Tensor, allowed: Tensor) -> Tensor:
    """Uniform gate; candidate 0 is UNKNOWN and cannot bridge two variables."""
    n, kmax, _ = validate(z, valid, allowed)
    out = torch.zeros_like(z)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for a in range(kmax):
                for b in range(kmax):
                    if not bool(valid[i, a] and valid[j, b]):
                        continue
                    terms = []
                    for k in range(n):
                        if k in (i, j) or not bool(allowed[i, j, k]):
                            continue
                        for c in range(1, kmax):
                            if bool(valid[k, c]):
                                terms.append(z[i, k, a, c] * z[k, j, c, b])
                    if terms:
                        out[i, j, a, b] = torch.stack(terms).mean(0)
    return out


def triangle_by_bridge(z: Tensor, valid: Tensor, allowed: Tensor) -> Tensor:
    """Same oracle, vectorized over targets and chunked over bridge variable."""
    n, _, _ = validate(z, valid, allowed)
    total = torch.zeros_like(z)
    counts = z.new_zeros(z.shape[:-1])
    ids = torch.arange(n, device=z.device)
    target_mask = valid[:, None, :, None] & valid[None, :, None, :]
    for k in range(n):
        bridge = valid[k].clone()
        bridge[0] = False
        path = (
            allowed[:, :, k]
            & (ids[:, None] != k)
            & (ids[None, :] != k)
            & (ids[:, None] != ids[None, :])
        )
        mask = target_mask & path[:, :, None, None]
        contribution = torch.einsum(
            "iacd,jcbd,c->ijabd", z[:, k], z[k], bridge.to(z.dtype)
        )
        total = total + contribution * mask[..., None].to(z.dtype)
        counts = counts + mask.to(z.dtype) * bridge.sum().to(z.dtype)
    return total / counts.clamp_min(1)[..., None]


def permute_candidates(z: Tensor, orders: Tensor) -> Tensor:
    return torch.stack([
        torch.stack([
            z[i, j].index_select(0, orders[i]).index_select(1, orders[j])
            for j in range(z.shape[1])
        ]) for i in range(z.shape[0])
    ])


class ReferenceTests(unittest.TestCase):
    def fixture(self):
        z = torch.zeros(3, 3, 3, 3, 1, dtype=torch.float64)
        valid = torch.ones(3, 3, dtype=torch.bool)
        allowed = torch.zeros(3, 3, 3, dtype=torch.bool)
        allowed[0, 2, 1] = True
        return z, valid, allowed

    def test_masked_softmax_empty(self):
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        p = masked_softmax(x, mask)
        self.assertTrue(bool(torch.isfinite(p).all()))
        torch.testing.assert_close(p.sum(-1), torch.tensor([1., 0.], dtype=x.dtype))
        self.assertEqual(p[0, 1].item(), 0.)

    def test_cross_candidate_does_not_connect(self):
        z, valid, allowed = self.fixture()
        z[0, 1, 1, 1] = 2.
        z[1, 2, 2, 1] = 3.
        self.assertEqual(triangle_reference(z, valid, allowed)[0, 2, 1, 1].item(), 0.)

    def test_matching_candidate_connects(self):
        z, valid, allowed = self.fixture()
        z[0, 1, 1, 1] = 2.
        z[1, 2, 1, 1] = 3.
        # Two valid bridge candidates: uniform mean of [6, 0] is 3.
        self.assertEqual(triangle_reference(z, valid, allowed)[0, 2, 1, 1].item(), 3.)

    def test_unknown_and_padding(self):
        z, valid, allowed = self.fixture()
        z[0, 1, 1, 0] = 100.
        z[1, 2, 0, 1] = 100.
        self.assertEqual(triangle_reference(z, valid, allowed).abs().sum().item(), 0.)
        z.fill_(1.)
        valid[0, 1] = False
        out = triangle_by_bridge(z, valid, allowed)
        self.assertEqual(out[0, 2, 1].abs().sum().item(), 0.)

    def test_empty_paths(self):
        z, valid, allowed = self.fixture()
        z.fill_(2.)
        valid[:, 1:] = False
        for fn in (triangle_reference, triangle_by_bridge):
            out = fn(z, valid, allowed)
            self.assertTrue(bool(torch.isfinite(out).all()))
            self.assertEqual(out.abs().sum().item(), 0.)

    def test_chunked_reference_agreement(self):
        torch.manual_seed(13)
        z = torch.randn(4, 4, 4, 4, 3, dtype=torch.float64)
        valid = torch.rand(4, 4) > .25
        valid[:, 0] = True
        allowed = torch.rand(4, 4, 4) > .3
        torch.testing.assert_close(
            triangle_by_bridge(z, valid, allowed),
            triangle_reference(z, valid, allowed), atol=1e-12, rtol=1e-12
        )

    def test_independent_candidate_permutation(self):
        torch.manual_seed(17)
        z = torch.randn(3, 3, 4, 4, 2, dtype=torch.float64)
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.bool)
        allowed = torch.ones(3, 3, 3, dtype=torch.bool)
        orders = torch.tensor([[0, 3, 1, 2], [0, 2, 3, 1], [0, 1, 3, 2]])
        expected = permute_candidates(triangle_reference(z, valid, allowed), orders)
        observed = triangle_reference(
            permute_candidates(z, orders), valid.gather(1, orders), allowed
        )
        torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-12)

    def test_gradients(self):
        torch.manual_seed(19)
        z = torch.randn(3, 3, 3, 3, 2, dtype=torch.float64, requires_grad=True)
        valid = torch.ones(3, 3, dtype=torch.bool)
        allowed = torch.ones(3, 3, 3, dtype=torch.bool)
        triangle_by_bridge(z, valid, allowed).square().sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(bool(torch.isfinite(z.grad).all()))
        self.assertGreater(z.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
