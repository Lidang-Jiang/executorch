# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.


import os
import unittest

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.datasets import (
    get_quant_test_data_super_resolution,
)
from executorch.backends.samsung.test.utils.quant_checkers import CheckerConfig
from executorch.backends.samsung.test.utils.utils import TestConfig
from executorch.examples.models.edsr import EdsrModel


class TestMilestoneEdsr(unittest.TestCase):
    def test_edsr_fp16(self):
        model = EdsrModel().get_eager_model()
        example_input = EdsrModel().get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=0.02)
        )

    def test_edsr_a8w8(self):
        example_input, cali, testdata = get_quant_test_data_super_resolution(
            os.path.join(os.environ["DATASET_PATH"]), "B100"
        )
        model = EdsrModel().get_eager_model()
        checker_config = CheckerConfig(
            "super_resolution", {"dataset": testdata, "threshold": 0.7}
        )
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )
        (
            tester.quantize(cali_dataset=cali, checker_config=checker_config)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=1, rtol=1)
        )
