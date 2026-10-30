# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import AttentionConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
    ReqMeta,
)
from vllm.forward_context import ForwardContext
from vllm.inputs import TextPrompt
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from .utils import create_vllm_config, make_two_group_large_page_config

MODEL_NAME = "RedHatAI/Qwen2.5-VL-3B-Instruct-quantized.w8a8"

SAMPLING_PARAMS = SamplingParams(temperature=0.0, top_k=1, max_tokens=128)

TEXT_PROMPTS = [
    "What's in the image(s)? Around 30 words. What's special in 2nd image?",
    "The future of AI is",
]


class InputCase(NamedTuple):
    text: str
    img: list[Image]
    expected_len: int
    info: str


def _check_path_len(path):
    """Return the latest length in path."""
    return len(list(path.iterdir()))


def _list_path(path):
    """Return the list of foldername (hashes generated) under the path."""
    return list(path.iterdir())


def run_test(
    tmp_path,
    processor,
    llm: LLM,
    question: str,
    image_urls: list[Image],
    expected_len: int,
    info: str,
):
    """One individual test to process the prompt and output base on 1 set of input
    Then check if the length in the storage path matches the expected length
    `info` introduces details or purpose of the individual test
    """
    print(f"***info: {info}***")
    print(f"**Expected storage path length after llm generate: {expected_len}**")
    process_prompt(processor, llm, question, image_urls)

    print(f"Path matched expected length: {_check_path_len(tmp_path)}")
    print(f"Hashes under the storage path: {_list_path(tmp_path)}")

    assert _check_path_len(tmp_path) == expected_len, (
        f"Expect storage path length {expected_len} ;",
        f"but end up {_check_path_len(tmp_path)} instead. ",
        f"Info: {info}",
    )


def process_prompt(processor, llm: LLM, question: str, image_urls: list[Image]):
    """Form the prompt based on the text and image input, then llm generate output."""
    placeholders = [
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(image_pil)},
        }
        for image_pil in image_urls
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_input: TextPrompt = {"prompt": prompt}
    if image_urls:
        prompt_input["multi_modal_data"] = {"image": image_urls}

    outputs = llm.generate(prompt_input, sampling_params=SAMPLING_PARAMS)

    print("-" * 50)
    print("Output:")
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


@pytest.mark.parametrize(
    "attn_backend",
    (
        ["FLASH_ATTN", "TRITON_ATTN"]
        if current_platform.is_cuda()
        else ["TRITON_ATTN"]
        if current_platform.is_rocm() or current_platform.is_xpu()
        else []
    ),
)
def test_shared_storage_connector_hashes(tmp_path, attn_backend):
    """Tests that ExampleConnector saves KV to the storage locations
    with proper hashes; that are unique for inputs with identical text but
    different images (same size), or same multiple images but different orders.
    """
    # Using tmp_path as the storage path to store KV
    print(f"KV storage path at: {str(tmp_path)}")

    # don't put this import at the top level
    # it will call torch.accelerator.device_count()
    from transformers import AutoProcessor

    # Create processor to handle the chat prompt
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Prepare images for the tests
    # Resize to the same size to check hashes correctness
    image_1 = ImageAsset("stop_sign").pil_image.resize((1280, 720))
    image_2 = ImageAsset("cherry_blossom").pil_image.resize((1280, 720))

    # Make sure that they are not the same picture
    assert image_1 != image_2, "The images should not be identical"

    # Create the LLM instance
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        attention_config=AttentionConfig(backend=attn_backend),
        enforce_eager=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
        limit_mm_per_prompt={"image": 2},
    )

    # Prepare the input cases
    input_cases = [
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=1,
            info="image_1 single input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the first time. "
                "It is in same pixel size with image_1, yet it "
                "should be able to form a new unique hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=2,
            info=(
                "image_1 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=3,
            info="image_1 with image_2 input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info="The image order is swapped. Should form new hash.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=4,
            info=(
                "[image_1, image_2] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info=(
                "[image_2, image_1] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Pure text input test as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Identical pure text input as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[1],
            img=[],
            expected_len=6,
            info="Another pure text input as a case-control",
        ),
    ]

    # Run tests
    for case_id, (text, img, expected_len, info) in enumerate(input_cases):
        print("\n", "=" * 25, f"Below running input case: {case_id}", "=" * 25)
        run_test(tmp_path, processor, llm, text, img, expected_len, info)

    print("All tests passed successfully!")


# ---------------------------------------------------------------------------
# CPU-only unit tests for the large-page block -> slot mapping.
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 16


def _dcp_like_kv_cache_config() -> KVCacheConfig:
    """Group 1's spec block size (8) differs from the global block size (16),
    as for DCP-sharded attention (B // dcp)."""
    return KVCacheConfig(
        num_pages=8,
        kv_cache_tensors=[],
        block_size=_BLOCK_SIZE,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa0"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
                num_blocks=8,
                blocks_per_page=1,
                kernel_rows_per_block=1,
                row_bytes=1024,
            ),
            KVCacheGroupSpec(
                ["dcp0"],
                FullAttentionSpec(
                    block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
                num_blocks=32,
                blocks_per_page=4,
                kernel_rows_per_block=1,
                row_bytes=256,
            ),
        ],
        large_page_bytes=1024,
    )


def test_make_meta_slot_mappings_use_each_groups_block_size():
    """A group whose spec block size differs from the global block size must
    get non-overlapping slot ranges that cover each block's rows exactly."""
    config = _dcp_like_kv_cache_config()
    meta = ReqMeta.make_meta(
        token_ids=list(range(33)),
        block_ids=([2, 3], [2, 3, 4, 5]),
        block_size=_BLOCK_SIZE,
        kv_cache_config=config,
        is_store=True,
        mm_hashes=[],
    )
    assert len(meta.token_ids) == 32
    # Group 0 (block size 16): blocks 2 and 3 cover slots 32-63.
    assert meta.slot_mappings[0].tolist() == list(range(32, 64))
    # Group 1 (block size 8): blocks 2-5 cover slots 16-47 with no overlap
    # or gap.
    assert meta.slot_mappings[1].tolist() == list(range(16, 48))


def test_example_connector_frees_blocks_immediately():
    """The connector is HMA-compatible and keeps no per-block state, so
    finished requests release their blocks in every group at once."""
    assert issubclass(ExampleConnector, SupportsHMA)
    connector = object.__new__(ExampleConnector)
    freed, kv_xfer_params = connector.request_finished_all_groups(
        SimpleNamespace(request_id="r0"), ([3, 5], [3, 5])
    )
    assert freed is False
    assert kv_xfer_params is None


def test_save_load_roundtrip_per_group_strided_views(tmp_path):
    """Store then load through per-layer strided views (the SWA view has the
    MLA shape H == 1): the load must restore exactly the rows the store read.
    """
    config = make_two_group_large_page_config()
    vllm_config = create_vllm_config(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
    )
    connector = ExampleConnector(vllm_config, KVConnectorRole.WORKER, config)

    # Per-layer strided views over each group's backing storage; row strides
    # are the group's row_bytes in float32 elements (2176 for FA, 256 for
    # SWA). FA has one kernel row per block (16 tokens), SWA four (4 tokens).
    fa_group, swa_group = config.kv_cache_groups
    fa_rows = fa_group.num_blocks
    swa_rows = swa_group.num_blocks * swa_group.kernel_rows_per_block
    fa_backing = torch.randn(fa_rows * 2176)
    fa_view = fa_backing.as_strided((fa_rows, 2, 16, 68), (2176, 1088, 68, 1))
    swa_backing = torch.randn(swa_rows * 256)
    swa_view = swa_backing.as_strided((swa_rows, 1, 4, 64), (256, 256, 64, 1))
    fa_orig = fa_backing.clone()
    swa_orig = swa_backing.clone()

    token_ids = list(range(33))
    block_ids = ([3, 5], [3, 5])
    store_meta = ExampleConnectorMetadata()
    store_meta.add_request(
        token_ids=token_ids,
        block_ids=block_ids,
        block_size=_BLOCK_SIZE,
        kv_cache_config=config,
        is_store=True,
        mm_hashes=[],
    )
    connector.bind_connector_metadata(store_meta)
    connector.save_kv_layer("fa.0", fa_view, attn_metadata=None)
    connector.save_kv_layer("swa.0", swa_view, attn_metadata=None)

    # Corrupt the rows the store read (FA rows 3 and 5, SWA rows 12-15 and
    # 20-23); the load must restore exactly those rows and nothing else.
    fa_view[3].zero_()
    fa_view[5].zero_()
    swa_view[12:16].zero_()
    swa_view[20:24].zero_()

    load_meta = ExampleConnectorMetadata()
    load_meta.add_request(
        token_ids=token_ids,
        block_ids=block_ids,
        block_size=_BLOCK_SIZE,
        kv_cache_config=config,
        is_store=False,
        mm_hashes=[],
    )
    connector.bind_connector_metadata(load_meta)
    forward_context = ForwardContext(
        no_compile_layers={
            "fa.0": SimpleNamespace(kv_cache=fa_view),
            "swa.0": SimpleNamespace(kv_cache=swa_view),
        },
        attn_metadata={"fa.0": None, "swa.0": None},
        slot_mapping={},
    )
    connector.start_load_kv(forward_context)

    assert torch.equal(fa_backing, fa_orig)
    assert torch.equal(swa_backing, swa_orig)
