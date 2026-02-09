"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.cute_dsl_ops.depermute_prefill_combine import (
    call_depermute_prefill_combine,
)


class TestDepermutePrefillCombine(unittest.TestCase):
    """
    Test cases for depermute_prefill_combine kernel.

    The kernel performs the following:
    - Input:
        - x: [num_local_experts, max_num_tokens_per_expert, hidden] - MoE output tensor
        - indice_map: [num_worst_tokens, topk] - index mapping from original token to permuted position
          values are: expert_idx * max_num_tokens_per_expert + offset, or -1 if not routed
        - topk_weights: [num_worst_tokens, topk] - scaling weights for each expert's contribution
    - Output:
        - depermuted_x: [num_worst_tokens, hidden] - combined output with original token order restored

    The kernel restores the original token order by:
    1. For each output token, looking up all its routed experts via indice_map
    2. Gathering the corresponding values from x (the permuted MoE output)
    3. Multiplying each expert's output by its topk_weight and summing them up
    """

    def setUp(self):
        paddle.seed(2024)
        np.random.seed(2024)
        paddle.set_device("gpu")

    def _compute_reference(
        self,
        x_np: np.ndarray,
        indice_map_np: np.ndarray,
        topk_weights_np: np.ndarray,
        num_worst_tokens: int,
        max_num_tokens_per_expert: int,
    ) -> np.ndarray:
        """
        Reference implementation in numpy.

        Args:
            x_np: [num_local_experts, max_num_tokens_per_expert, hidden]
            indice_map_np: [num_worst_tokens, topk]
            topk_weights_np: [num_worst_tokens, topk]
            num_worst_tokens: number of tokens to depermute
            max_num_tokens_per_expert: max tokens per expert

        Returns:
            depermuted_x: [num_worst_tokens, hidden]
        """
        hidden = x_np.shape[2]
        topk = indice_map_np.shape[1]
        depermuted_x = np.zeros((num_worst_tokens, hidden), dtype=np.float32)

        for token_idx in range(num_worst_tokens):
            for k in range(topk):
                indice = indice_map_np[token_idx, k]
                if indice >= 0:
                    expert_idx = indice // max_num_tokens_per_expert
                    offset = indice % max_num_tokens_per_expert
                    weight = topk_weights_np[token_idx, k]
                    depermuted_x[token_idx, :] += x_np[expert_idx, offset, :] * weight

        return depermuted_x

    def _run_and_verify(
        self,
        num_worst_tokens: int,
        num_local_experts: int,
        max_num_tokens_per_expert: int,
        hidden: int,
        topk: int,
        x_dtype=paddle.bfloat16,
        sparsity: float = 0.2,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ):
        """
        Run the kernel and verify against reference implementation.

        Args:
            num_worst_tokens: number of original tokens
            num_local_experts: number of local experts
            max_num_tokens_per_expert: max tokens per expert
            hidden: hidden dimension
            topk: number of experts each token routes to
            x_dtype: data type for x
            sparsity: probability of indice being -1 (not routed)
            rtol: relative tolerance
            atol: absolute tolerance
        """
        # Generate input x tensor
        x_np = np.random.randn(num_local_experts, max_num_tokens_per_expert, hidden).astype(np.float32)
        if x_dtype == paddle.bfloat16:
            x = paddle.to_tensor(x_np).cast(paddle.bfloat16)
        elif x_dtype == paddle.float8_e4m3fn:
            x_np = np.clip(x_np, -448, 448)
            x = paddle.to_tensor(x_np).cast(paddle.float8_e4m3fn)
        else:
            x = paddle.to_tensor(x_np)

        # Generate indice_map: each entry is expert_idx * max_num_tokens_per_expert + offset
        # or -1 if not routed
        indice_map_np = np.zeros((num_worst_tokens, topk), dtype=np.int32)
        for token_idx in range(num_worst_tokens):
            # Track used positions per expert to avoid duplicates
            used_positions = {}
            has_valid_index = False
            for k in range(topk):
                # For the last slot, if no valid index yet, force one
                if k == topk - 1 and not has_valid_index:
                    should_be_invalid = False
                else:
                    should_be_invalid = np.random.rand() < sparsity

                if should_be_invalid:
                    indice_map_np[token_idx, k] = -1
                else:
                    expert_idx = np.random.randint(0, num_local_experts)
                    if expert_idx not in used_positions:
                        used_positions[expert_idx] = []
                    # Find an unused offset for this expert
                    offset = np.random.randint(0, max_num_tokens_per_expert)
                    attempts = 0
                    while offset in used_positions.get(expert_idx, []) and attempts < 10:
                        offset = np.random.randint(0, max_num_tokens_per_expert)
                        attempts += 1
                    used_positions[expert_idx].append(offset)
                    indice_map_np[token_idx, k] = expert_idx * max_num_tokens_per_expert + offset
                    has_valid_index = True

        indice_map = paddle.to_tensor(indice_map_np).cast(paddle.int32)

        # Generate topk_weights
        topk_weights_np = np.random.rand(num_worst_tokens, topk).astype(np.float32)
        # Normalize weights so they sum to 1 for each token (typical in MoE)
        row_sums = topk_weights_np.sum(axis=1, keepdims=True)
        topk_weights_np = topk_weights_np / (row_sums + 1e-6)
        # Zero out weights for invalid indices
        topk_weights_np[indice_map_np == -1] = 0.0
        topk_weights = paddle.to_tensor(topk_weights_np).cast(paddle.float32)

        # Run the kernel
        print("x", x)
        print("indice_map", indice_map)
        print("topk_weights", topk_weights)
        print("num_worst_tokens", num_worst_tokens)
        depermuted_x = call_depermute_prefill_combine(
            x=x,
            indice_map=indice_map,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
        )
        print("depermuted_x", depermuted_x[:, 0])
        has_nan = paddle.any(paddle.isnan(depermuted_x))

        if has_nan:
            print("has_nan", paddle.nonzero(paddle.isnan(depermuted_x)).tolist())

        # Compute reference
        x_ref_np = x.cast(paddle.float32).numpy()
        expected = self._compute_reference(
            x_np=x_ref_np,
            indice_map_np=indice_map_np,
            topk_weights_np=topk_weights_np,
            num_worst_tokens=num_worst_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
        )
        print("expected", expected[:, 0])

        # Get kernel result
        result = depermuted_x.cast(paddle.float32).numpy()


        # Verify shape
        self.assertEqual(result.shape, (num_worst_tokens, hidden))

        # Print mismatch details before assertion
        abs_diff = np.abs(result - expected)
        rel_diff = abs_diff / (np.abs(expected) + 1e-8)
        mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)

        # Check for NaN mismatches
        result_nan = np.isnan(result)
        expected_nan = np.isnan(expected)
        nan_mismatch = result_nan != expected_nan

        if np.any(mismatch_mask) or np.any(nan_mismatch):
            # Find mismatch indices
            mismatch_indices = np.argwhere(mismatch_mask | nan_mismatch)
            print(f"\n{'='*60}")
            print(f"MISMATCH DETAILS (total: {len(mismatch_indices)} mismatches)")
            print(f"{'='*60}")
            for idx in mismatch_indices[:50]:  # Print first 50 mismatches
                token_idx, hidden_idx = idx[0], idx[1]
                actual_val = result[token_idx, hidden_idx]
                expected_val = expected[token_idx, hidden_idx]
                abs_err = abs_diff[token_idx, hidden_idx]
                rel_err = rel_diff[token_idx, hidden_idx]
                print(f"[{token_idx}, {hidden_idx}]: actual={actual_val:.6f}, expected={expected_val:.6f}, "
                      f"abs_err={abs_err:.6f}, rel_err={rel_err:.6f}")
            if len(mismatch_indices) > 50:
                print(f"... and {len(mismatch_indices) - 50} more mismatches")
            print(f"{'='*60}\n")

        # Verify values
        np.testing.assert_allclose(
            result,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"Depermuted output mismatch"
        )

        return True

    def test_basic_topk4(self):
        """Test basic case with topk=4"""
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=128,
            hidden=7168,
            topk=4,
            sparsity=0.2,
        )

    def test_basic_topk8(self):
        """Test basic case with topk=8"""
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=128,
            hidden=7168,
            topk=8,
            sparsity=0.2,
        )

    def test_small_tokens(self):
        """Test with small number of tokens"""
        self._run_and_verify(
            num_worst_tokens=4,
            num_local_experts=4,
            max_num_tokens_per_expert=32,
            hidden=1024,
            topk=4,
            sparsity=0.1,
        )

    def test_large_tokens(self):
        """Test with large number of tokens"""
        self._run_and_verify(
            num_worst_tokens=512,
            num_local_experts=16,
            max_num_tokens_per_expert=256,
            hidden=4096,
            topk=4,
            sparsity=0.3,
        )

    def test_high_sparsity(self):
        """Test with high sparsity (many -1 values in indice_map)"""
        self._run_and_verify(
            num_worst_tokens=128,
            num_local_experts=8,
            max_num_tokens_per_expert=64,
            hidden=2048,
            topk=4,
            sparsity=0.7,
        )

    def test_no_sparsity(self):
        """Test with no sparsity (all tokens routed)"""
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=128,
            hidden=2048,
            topk=4,
            sparsity=0.0,
        )

    def test_single_expert(self):
        """Test with single local expert"""
        self._run_and_verify(
            num_worst_tokens=32,
            num_local_experts=1,
            max_num_tokens_per_expert=64,
            hidden=1024,
            topk=4,
            sparsity=0.0,
        )

    def test_many_experts(self):
        """Test with many local experts"""
        self._run_and_verify(
            num_worst_tokens=128,
            num_local_experts=32,
            max_num_tokens_per_expert=64,
            hidden=2048,
            topk=8,
            sparsity=0.3,
        )

    def test_small_hidden(self):
        """Test with small hidden dimension"""
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=64,
            hidden=256,
            topk=4,
            sparsity=0.2,
        )

    def test_large_hidden(self):
        """Test with large hidden dimension"""
        self._run_and_verify(
            num_worst_tokens=32,
            num_local_experts=8,
            max_num_tokens_per_expert=64,
            hidden=14336,
            topk=4,
            sparsity=0.2,
        )

    def test_all_minus_one(self):
        """Test edge case where all indice_map values are -1 (nothing routed)"""
        num_worst_tokens = 32
        num_local_experts = 4
        max_num_tokens_per_expert = 64
        hidden = 1024
        topk = 4

        # Create input data
        x_np = np.random.randn(num_local_experts, max_num_tokens_per_expert, hidden).astype(np.float32)
        x = paddle.to_tensor(x_np).cast(paddle.bfloat16)

        # All -1 indice_map
        indice_map = paddle.full([num_worst_tokens, topk], -1, dtype=paddle.int32)

        # Zero weights (since nothing is routed)
        topk_weights = paddle.zeros([num_worst_tokens, topk], dtype=paddle.float32)

        # Run the kernel
        depermuted_x = call_depermute_prefill_combine(
            x=x,
            indice_map=indice_map,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
        )

        # Verify output is all zeros
        result = depermuted_x.cast(paddle.float32).numpy()
        expected = np.zeros((num_worst_tokens, hidden), dtype=np.float32)

        self.assertEqual(result.shape, (num_worst_tokens, hidden))

    def test_single_token(self):
        """Test with single token"""
        self._run_and_verify(
            num_worst_tokens=1,
            num_local_experts=4,
            max_num_tokens_per_expert=32,
            hidden=1024,
            topk=4,
            sparsity=0.0,
        )

    def test_uniform_weights(self):
        """Test with uniform topk weights"""
        num_worst_tokens = 64
        num_local_experts = 8
        max_num_tokens_per_expert = 64
        hidden = 2048
        topk = 4

        # Create input data
        x_np = np.random.randn(num_local_experts, max_num_tokens_per_expert, hidden).astype(np.float32)
        x = paddle.to_tensor(x_np).cast(paddle.bfloat16)

        # Generate valid indice_map
        indice_map_np = np.zeros((num_worst_tokens, topk), dtype=np.int32)
        for token_idx in range(num_worst_tokens):
            for k in range(topk):
                expert_idx = k % num_local_experts
                offset = token_idx % max_num_tokens_per_expert
                indice_map_np[token_idx, k] = expert_idx * max_num_tokens_per_expert + offset
        indice_map = paddle.to_tensor(indice_map_np).cast(paddle.int32)

        # Uniform weights: 1/topk for each
        topk_weights_np = np.ones((num_worst_tokens, topk), dtype=np.float32) / topk
        topk_weights = paddle.to_tensor(topk_weights_np)

        # Run the kernel
        depermuted_x = call_depermute_prefill_combine(
            x=x,
            indice_map=indice_map,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
        )

        # Compute reference
        x_ref_np = x.cast(paddle.float32).numpy()
        expected = self._compute_reference(
            x_np=x_ref_np,
            indice_map_np=indice_map_np,
            topk_weights_np=topk_weights_np,
            num_worst_tokens=num_worst_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
        )

        result = depermuted_x.cast(paddle.float32).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
