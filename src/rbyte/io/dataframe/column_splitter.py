from typing import Any
import polars as pl
from typing import final
from optree import PyTree, tree_map

@final
class DataFrameColumnSplitter:
    __name__ = __qualname__

    def __init__(self, columns: dict[str, dict[str, Any]]):
        self.columns = columns

    def __call__(self, input: PyTree[pl.DataFrame]) -> PyTree[pl.DataFrame]:
        return tree_map(self._process_dataframe, input)

    def _process_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        for in_col_name, out_col_info in self.columns.items():
            col_series = df[in_col_name]  # Store the column series for reuse
            assert col_series.dtype == pl.List, (
                f"Column `{in_col_name}` is of type {col_series.dtype}, expected {pl.List}"
            )
            lengths = col_series.list.len()
            assert lengths.min() == lengths.max() == len(out_col_info['out_columns']), (
                f"Column `{in_col_name}` has lists of different lengths"
            )
            out_columns = out_col_info['out_columns']
            df = df.with_columns(
                *[
                    col_series
                    .explode()
                    .gather_every(n=len(out_columns), offset=i)
                    .alias(c)
                    .cast(out_col_info['out_dtype'])
                    for i, c in enumerate(out_columns)
                ]
            )
            if out_col_info.get('drop_column', False):
                df = df.drop(in_col_name)
        return df