import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skl2onnx.proto import onnx_proto
from skl2onnx.common._apply_operation import (
    apply_cast,
    apply_concat,
)
from skl2onnx import update_registered_converter
from skl2onnx.common._topology import Scope, Operator
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    DoubleTensorType,
)
from skl2onnx.common.utils import check_input_and_output_numbers
from skl2onnx.common.utils import check_input_and_output_types


class CumulativeBinningTransformer(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_bins=5,
    ):
        self.n_bins = n_bins

    def fit(self, X, y=None):

        X = self._validate_data(X, dtype="numeric", force_all_finite=False)

        n_features = X.shape[1]
        n_bins = np.full((n_features,), fill_value=self.n_bins)

        bin_edges = np.zeros(n_features, dtype=object)

        for jj in range(n_features):
            column = X[:, jj]
            column = column[~np.isnan(column)]
            col_min, col_max = column.min(), column.max()

            if col_min == col_max:
                warnings.warn(f"Feature {jj} is constant and will be replaced with 0.")
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            quantiles = np.linspace(0, 100, n_bins[jj] + 1)
            bin_edges[jj] = np.asarray(np.percentile(column, quantiles))

            bin_edges[jj][0] = -np.inf
            bin_edges[jj][-1] = np.inf

            mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
            bin_edges[jj] = bin_edges[jj][mask]
            if len(bin_edges[jj]) - 1 != n_bins[jj]:
                warnings.warn(
                    f"Bins whose width are too small (i.e., <= 1e-8) in feature {jj} are removed. Consider decreasing the number of bins."
                )
                n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        return self

    def transform(self, X):

        check_is_fitted(self)

        Xt = self._validate_data(
            X,
            copy=True,
            dtype=(np.float64, np.float32),
            reset=False,
            force_all_finite=False,
        )

        bin_edges = self.bin_edges_
        columns = []

        for jj in range(Xt.shape[1]):
            column = Xt[:, [jj]] > bin_edges[jj][:-1]  # right-closed
            columns.append(column)

        return np.concatenate(columns, axis=1, dtype=np.float32)


def cumulative_binning_transformer_shape_calculator(operator: Operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    M = operator.inputs[0].get_first_dimension()
    model = operator.raw_operator
    N = sum(model.n_bins_)
    operator.outputs[0].type.shape = [M, N]


def cumulative_binning_transformer_shape_converter(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator

    ranges = list(
        map(
            lambda e: e[:-1] if len(e) > 1 else [np.finfo(np.float32).max],
            op.bin_edges_,
        )
    )
    digitised_output_name = [None] * len(ranges)

    for i, item in enumerate(ranges):
        digitised_output_name[i] = scope.get_unique_variable_name(
            "digitised_output_{}".format(i)
        )
        column_index_name = scope.get_unique_variable_name("column_index")
        range_column_name = scope.get_unique_variable_name("range_column")
        column_name = scope.get_unique_variable_name("column")
        cast_column_name = scope.get_unique_variable_name("cast_column")
        greater_result_name = scope.get_unique_variable_name("greater_result")

        container.add_initializer(
            column_index_name, onnx_proto.TensorProto.INT64, [], [i]
        )
        container.add_initializer(
            range_column_name, onnx_proto.TensorProto.FLOAT, [len(item)], item
        )

        container.add_node(
            "ArrayFeatureExtractor",
            [operator.inputs[0].full_name, column_index_name],
            column_name,
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
            op_domain="ai.onnx.ml",
        )
        apply_cast(
            scope,
            column_name,
            cast_column_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        container.add_node(
            "Greater",
            [cast_column_name, range_column_name],
            greater_result_name,
            name=scope.get_unique_operator_name("Greater"),
        )
        apply_cast(
            scope,
            greater_result_name,
            digitised_output_name[i],
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
    apply_concat(
        scope, digitised_output_name, operator.outputs[0].full_name, container, axis=1
    )


update_registered_converter(
    CumulativeBinningTransformer,
    "CumulativeBinningTransformer",
    cumulative_binning_transformer_shape_calculator,
    cumulative_binning_transformer_shape_converter,
)
