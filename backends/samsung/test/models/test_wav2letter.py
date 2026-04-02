# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.
import os
import unittest

import torch
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.datasets import get_quant_test_data_voice
from executorch.backends.samsung.test.utils.quant_checkers import CheckerConfig
from executorch.backends.samsung.test.utils.utils import TestConfig
from executorch.examples.models.wav2letter import Wav2LetterModel


class TestMilestoneWav2Letter(unittest.TestCase):
    def test_w2l_fp16(self):
        model = Wav2LetterModel().get_eager_model()
        example_input = Wav2LetterModel().get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=0.009)
        )

    def test_w2l_quant(self):
        factory = Wav2LetterModel()
        factory.vocab_size = 29
        assert (model_cache_dir := os.getenv("MODEL_CACHE")), "MODEL_CACHE not set!"
        weight_path = os.path.join(model_cache_dir, "w2l/states_fused.pth")
        state_dict = torch.load(weight_path, weights_only=False)
        model = factory.get_eager_model()
        model.load_state_dict(state_dict)
        example_input, calib_data, quant_test_data = get_quant_test_data_voice(
            os.path.join(os.environ["DATASET_PATH"], "w2l/wav2letter")
        )
        labels = [" ", *"abcdefghijklmnopqrstuvwxyz", "'", "*"]
        checker_config = CheckerConfig(
            "wave2letter", {"dataset": quant_test_data, "labels": labels}
        )
        (
            SamsungTester(
                model,
                example_input,
                [gen_samsung_backend_compile_spec(TestConfig.chipset)],
            )
            .quantize(cali_dataset=calib_data, checker_config=checker_config)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(atol=1.0, rtol=1.0)
        )
