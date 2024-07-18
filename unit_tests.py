import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from cumulative_binning_transformer import CumulativeBinningTransformer


x = np.array(
    [[np.nan, np.inf], [1.0, np.nan], [np.nan, -np.inf], [42, 3.14]],
    dtype=np.float32,
)

transformer = CumulativeBinningTransformer(n_bins=5).fit(x)

initial_type = [("float_input", FloatTensorType([None, transformer.n_features_in_]))]

transformer_onnx = convert_sklearn(transformer, initial_types=initial_type)

x_transformed = transformer.transform(x)

session = ort.InferenceSession(transformer_onnx.SerializeToString())
x_transformed_onnx = session.run(None, {"float_input": x})[0]

def test_unit_n_bins_vs_bin_edges_consistency() -> None:
    for n_bins, bin_edges in zip(transformer.n_bins_, transformer.bin_edges_):
        assert n_bins + 1 == len(bin_edges)


def test_unit_sklearn_vs_onnx_consistency() -> None:
    assert (x_transformed == x_transformed_onnx).all()


def test_infinite_infinite_transforms_to_zero_row() -> None:
    assert (x_transformed_onnx[2, :] == 0).all()
