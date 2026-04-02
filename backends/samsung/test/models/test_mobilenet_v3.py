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
from executorch.backends.samsung.test.utils.datasets import get_quant_test_data_classify
from executorch.backends.samsung.test.utils.quant_checkers import CheckerConfig
from executorch.backends.samsung.test.utils.utils import TestConfig
from executorch.examples.models.mobilenet_v3 import MV3Model


class TestMilestoneMobilenetV3(unittest.TestCase):
    def test_mv3_fp16(self):
        torch.manual_seed(8)
        model = MV3Model().get_eager_model()
        example_input = MV3Model().get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=0.07, rtol=0.07)
        )

    def test_mv3_a8w8(self):
        example_input, cali, testdata = get_quant_test_data_classify(
            os.path.join(os.environ["DATASET_PATH"], "imagenet_ptq_subset")
        )
        checker_config = CheckerConfig(
            "classifier",
            {
                "dataset": testdata,
                "topktol": {1: 0.0, 2: 0.0},
            },
        )
        model = MV3Model().get_eager_model()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )
        (
            tester.quantize(cali_dataset=cali, checker_config=checker_config)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=3, rtol=3)
        )
