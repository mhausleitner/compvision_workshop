import os.path
import types
import unittest
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec

import numpy as np
import numpy.testing as npt


def import_from_source(name: str, file_path: str) -> types.ModuleType:
    loader: SourceFileLoader = SourceFileLoader(name, file_path)
    spec: ModuleSpec = spec_from_loader(loader.name, loader)
    module: types.ModuleType = module_from_spec(spec)
    loader.exec_module(module)
    return module


script_path: str = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "transform_image_inputs.py",
    )
)

under_test: types.ModuleType = import_from_source("transform_image_inputs", script_path)

input_polygon_coords = [[303.020119859749, 283.054324183695], [272.811437679515, 283.054324183695],
                        [272.811437679515, 310.179385725175], [303.0201198597498, 310.179385725175]]

expected_yolo_bb_str = '0.4490871543275507 0.5783532323328808 0.04876356590661688 0.05297863582320317'

expected_yolo_bb_array = [0.449087154327550, 0.578353232332881, 0.0487635659066168, 0.0529786]


class bb_to_yolo_tests(unittest.TestCase):
    """
    Assure boundingbox calculation returns correct result
    For this test we took the first label from our dataset
    """

    def test_get_bounding_box(self):
        expected = np.zeros((2, 2))
        expected[0, 0] = 272.811437679515
        expected[0, 1] = 283.054324183695
        expected[1, 0] = 303.020119859749
        expected[1, 1] = 310.179385725175
        result = under_test.get_bounding_box(input_polygon_coords)[0]
        npt.assert_array_almost_equal(expected, result)  # use almost because the strict equals method rounds

    """
    Assure the result for a coords 2 yolo-friendly boundingbox transformation
    """

    def test_calculate_yolobb(self):
        input = np.zeros((2, 2))
        input[0, 0] = 272.811437679515
        input[0, 1] = 283.054324183695
        input[1, 0] = 303.020119859749
        input[1, 1] = 310.179385725175
        result = under_test.calculate_yolobb(input)
        npt.assert_almost_equal(expected_yolo_bb_array, result)

    """
    Assure result from normalizing the coords
    Calls calculate_yolobb in method
    """

    def test_normalize_coords(self):
        result = under_test.normalize_coords(input_polygon_coords)
        self.assertEqual(expected_yolo_bb_str, result)


if __name__ == '__main__':
    unittest.main()
