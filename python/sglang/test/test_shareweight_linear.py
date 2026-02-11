import json
import os
import unittest
from typing import List

import torch
from safetensors.torch import safe_open
from unittest.mock import MagicMock, patch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.test.test_utils import CustomTestCase


class TestShareWeightLinear(CustomTestCase):
    """测试 ColumnParallelLinear + Fp8LinearMethod 的 split prefill 共享权重切分逻辑。

    - 使用 DeepSeek-V3.1-mini 的实际 FP8 权重
      `model.layers.0.self_attn.q_b_proj.weight` 作为底层权重。
    - 通过 mock 分布式环境，让 ColumnParallelLinear 在 TP=1 下初始化，
      然后在 `apply_for_split_prefill` 中人为传入 `tp_size/tp_rank/tp_attention_size`，
      验证切分行为是否等价于对 decode 块手工按 rank 切分。
    """

    MODEL_DIR = "/sgl-workspace/data/DeepSeek-V3.1-mini"
    INDEX_JSON = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    TARGET_WEIGHT_NAME = "model.layers.0.self_attn.q_b_proj.weight"

    _patches: List[patch] = []

    @classmethod
    def setUpClass(cls):
        # 如果权重目录不存在，直接跳过整个测试
        if not os.path.isdir(cls.MODEL_DIR) or not os.path.isfile(cls.INDEX_JSON):
            raise unittest.SkipTest(
                f"DeepSeek-V3.1-mini weights not found at {cls.MODEL_DIR}"
            )

        # mock TP 环境：TP=1，避免真正初始化 distributed
        tp_group = MagicMock()
        tp_group.world_size = 1
        tp_group.rank_in_group = 0

        cls._patches = [
            # 源头 parallel_state
            patch(
                "sglang.srt.distributed.parallel_state.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.model_parallel_is_initialized",
                return_value=True,
            ),
            # re-export 的 distributed 接口
            patch(
                "sglang.srt.distributed.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.distributed.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
            # linear 模块内直接导入的 get_tp_group 也需要 patch
            patch("sglang.srt.layers.linear.get_tp_group", return_value=tp_group),
        ]
        for p in cls._patches:
            p.start()

        # 读取 quantization_config，构造 Fp8Config
        with open(os.path.join(cls.MODEL_DIR, "config.json"), "r") as f:
            cfg = json.load(f)
        quant_cfg = cfg.get("quantization_config", None)
        if quant_cfg is None:
            raise unittest.SkipTest("quantization_config not found in DeepSeek config")
        cls.fp8_config = Fp8Config.from_config(quant_cfg)

        # 从 index.json 中找到目标权重所在的 safetensors 文件
        with open(cls.INDEX_JSON, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", index)
        if cls.TARGET_WEIGHT_NAME not in weight_map:
            raise unittest.SkipTest(
                f"{cls.TARGET_WEIGHT_NAME} not found in model.safetensors.index.json"
            )
        shard_file = weight_map[cls.TARGET_WEIGHT_NAME]
        cls.weight_path = os.path.join(cls.MODEL_DIR, shard_file)

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()

    def _load_fp8_weight(self):
        """从 safetensors 文件中读取一个 FP8 权重张量。"""
        with safe_open(self.weight_path, framework="pt") as f:
            tensor = f.get_tensor(self.TARGET_WEIGHT_NAME)
        print(tensor.shape)
        return tensor

    def test_fp8_split_prefill_on_dummy_layer(self):
        """在一个 Dummy layer 上直接测试 Fp8LinearMethod 的 split prefill 切分逻辑。

        这样可以避开 linear/auto_round 的循环导入，同时验证
        `apply_for_split_prefill` 的两段划分是否正确。
        """
        weight_tensor = self._load_fp8_weight()  # [out_dim, in_dim]
        out_dim, in_dim = weight_tensor.shape

        class DummyLayer(torch.nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()
                self.input_size_per_partition = in_size
                self.output_size_per_partition = out_size
                # 这里只存 decode 块，直接用 checkpoint 的一整块
                self.weight = torch.nn.Parameter(
                    torch.empty(out_size, in_size, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                # block quant 情况下需要 weight_scale_inv，shape 与 block 网格对应
                block_n, block_k = self.quant_method.quant_config.weight_block_size
                self.weight_scale_inv = torch.nn.Parameter(
                    torch.empty(
                        (out_size + block_n - 1) // block_n,
                        (in_size + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.bias = None

        # 构造 quant_method 和 dummy layer
        qm = Fp8LinearMethod(self.fp8_config)
        DummyLayer.quant_method = qm  # 简单挂到类上，方便 __init__ 使用
        layer = DummyLayer(in_dim, out_dim)

        # 将 checkpoint 权重直接拷入 layer.weight
        layer.weight.data.copy_(weight_tensor)
        # 为了测试切分，只要 scale 是合法形状即可，具体数值无关紧要
        layer.weight_scale_inv.data.zero_()

        # mock 掉底层 FP8 kernel，使其返回 weight_shard，方便观察切分结果
        import sglang.srt.layers.quantization.fp8 as fp8_mod

        def fake_block_fp8_linear(input, weight, block_size, weight_scale, input_scale, bias):
            # 直接返回 weight_shard，忽略真正 GEMM
            return weight

        qm.w8a8_block_fp8_linear = fake_block_fp8_linear
        fp8_mod.use_intel_amx_backend = lambda layer: False  # 避免走 AMX 分支

        # 设定一个简单的 prefill 场景：tp_size=4, tp_attention_size=2
        tp_size = 4
        tp_attention_size = 2
        tp_attention_rank = 0  # 由于当前 layer 视角是“一个 decode 块”，这里值只用于 local_size 计算

        # local_prefill_size = tp_size // tp_attention_size = 2
        local_prefill_size = tp_size // tp_attention_size
        out_chunk = layer.output_size_per_partition
        shard_n = out_chunk // local_prefill_size

        # 我们只检查 tp_rank=0 和 1，验证在 decode 块内的第二次划分
        for tp_rank in (0, 1):
            # 随便造一个 input，kernel 会被 fake 掉，不影响切分
            x = torch.zeros(1, in_dim, dtype=torch.bfloat16)
            out = qm.apply_for_split_prefill(
                layer=layer,
                x=x,
                bias=None,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_attention_size=tp_attention_size,
                tp_attention_rank=tp_attention_rank,
                is_column=True,
            )
            # out 实际上就是 weight_shard: [shard_n, in_dim]
            self.assertEqual(out.shape, (shard_n, in_dim))

            local_rank = tp_rank % local_prefill_size
            start_n = local_rank * shard_n

            # 预期切分：在 decode 块（此处即整块）里按上述公式切 rows
            expected = weight_tensor[start_n : start_n + shard_n, :]

            self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_column_parallel_linear_split_prefill(self):
        """在真实 ColumnParallelLinear 上测试 split prefill + Fp8LinearMethod 的 glue。"""
        from sglang.srt.layers.linear import ColumnParallelLinear
        import sglang.srt.layers.quantization.fp8 as fp8_mod

        weight_tensor = self._load_fp8_weight()  # [out_full, in_dim]
        out_full, in_dim = weight_tensor.shape

        tp_size = 4
        tp_attention_size = 2
        tp_attention_rank = 0

        # 只取一个 decode 块 [0,1,...]，大小为 out_full / tp_attention_size
        out_chunk = out_full // tp_attention_size

        lin = ColumnParallelLinear(
            input_size=in_dim,
            output_size=out_chunk,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=torch.bfloat16,
            quant_config=self.fp8_config,
        )

        # 确认 quant_method 为 Fp8LinearMethod 且是 block quant
        self.assertIsInstance(lin.quant_method, Fp8LinearMethod)
        self.assertTrue(lin.quant_method.block_quant)

        # 用 checkpoint 的前 out_chunk 行覆盖 decode 块权重
        with torch.no_grad():
            lin.weight.data.copy_(weight_tensor[0:out_chunk, :])

        # fake 掉 FP8 kernel，让 apply_for_split_prefill 返回 weight_shard
        def fake_block_fp8_linear(input, weight, block_size, weight_scale, input_scale, bias):
            return weight

        lin.quant_method.w8a8_block_fp8_linear = fake_block_fp8_linear
        fp8_mod.use_intel_amx_backend = lambda layer: False

        local_prefill_size = tp_size // tp_attention_size  # 块内 prefill rank 数
        shard_n = out_chunk // local_prefill_size

        for tp_rank in (0, 1):
            x = torch.zeros(1, in_dim, dtype=torch.bfloat16)
            out = lin.forward_split_prefill(
                x,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_attention_size=tp_attention_size,
                tp_attention_rank=tp_attention_rank,
            )
            # fake kernel 返回的是 weight_shard: [shard_n, in_dim]
            self.assertEqual(out.shape, (shard_n, in_dim))

            local_rank = tp_rank % local_prefill_size
            start_n = local_rank * shard_n
            expected_block = weight_tensor[0:out_chunk, :]
            expected = expected_block[start_n : start_n + shard_n, :]

            self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_row_parallel_linear_split_prefill(self):
        """在真实 RowParallelLinear 上测试 split prefill + Fp8LinearMethod 的 glue。"""
        from sglang.srt.layers.linear import RowParallelLinear
        import sglang.srt.layers.quantization.fp8 as fp8_mod

        weight_tensor = self._load_fp8_weight()  # [out_dim, in_full]
        out_dim, in_full = weight_tensor.shape

        tp_size = 4
        tp_attention_size = 2
        tp_attention_rank = 0

        lin = RowParallelLinear(
            input_size=in_full,
            output_size=out_dim,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=torch.bfloat16,
            reduce_results=False,
            quant_config=self.fp8_config,
        )

        self.assertIsInstance(lin.quant_method, Fp8LinearMethod)
        self.assertTrue(lin.quant_method.block_quant)

        with torch.no_grad():
            lin.weight.data.copy_(weight_tensor)

        def fake_block_fp8_linear(input, weight, block_size, weight_scale, input_scale, bias):
            return weight

        lin.quant_method.w8a8_block_fp8_linear = fake_block_fp8_linear
        fp8_mod.use_intel_amx_backend = lambda layer: False

        local_prefill_size = tp_size // tp_attention_size
        shard_k = in_full // local_prefill_size

        for tp_rank in (0, 1):
            x = torch.zeros(1, in_full, dtype=torch.bfloat16)
            out = lin.forward_split_prefill(
                x,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_attention_size=tp_attention_size,
                tp_attention_rank=tp_attention_rank,
            )

            # fake kernel 返回的是 weight_shard: [out_dim, shard_k]
            self.assertEqual(out.shape, (out_dim, shard_k))

            local_rank = tp_rank % local_prefill_size
            start_k = local_rank * shard_k
            expected = weight_tensor[:, start_k : start_k + shard_k]

            self.assertTrue(torch.equal(out.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()

