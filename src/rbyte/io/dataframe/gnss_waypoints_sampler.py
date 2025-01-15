from typing import final

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


# TODO:
# - [ ] add pydantic config


@final
class DataFrameGnssWaypointsSampler:
    __name__ = __qualname__

    def __init__(
        self,
        columns: dict[str, str],
        num_waypoints: int = 10,
        time_window_seconds: int = 60,
        radius_meters: int = 200,
        heading_error_thr_deg: int = 10,
        heading_smoothing_window: int = 15,
    ) -> None:
        self.columns: dict[str, str] = columns
        self.num_waypoints: int = num_waypoints
        self.time_window_seconds: int = time_window_seconds
        self.radius_degrees: float = self._approximate_radius_deg(radius_meters)
        self.heading_error_thr_deg = heading_error_thr_deg
        self.heading_smoothing_window = heading_smoothing_window

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        # TODO: brush up and make proper naming
        # WARN: Currently it's only forward looking
        ts_col: str = self.columns["time_stamp"]
        lat_col: str = self.columns["latitude"]
        lon_col: str = self.columns["longitude"]
        input[0, self.columns["heading_error"]] = 0
        # find waypoints in time_window_seconds radius
        logger.debug("finding waypoints in time_window_seconds radius")
        df = (
            input.rolling(
                ts_col,
                # offset=f"-{int(self.time_window_seconds / 2)}s",
                offset="0s",
                period=f"{self.time_window_seconds}s",
            )
            .agg([pl.struct([lat_col, lon_col]).alias("rolling_lat_lon")])
            .explode("rolling_lat_lon")
            .unnest("rolling_lat_lon")
            .rename({lat_col: "rolling_lat", lon_col: "rolling_lon"})
            .join(input, on=ts_col, how="left")
        )

        logger.debug("filtering by distance radius")
        # filter by distance radius
        df = (
            df.filter(
                (pl.col("rolling_lat") - pl.col(lat_col)) ** 2
                + (pl.col("rolling_lon") - pl.col(lon_col)) ** 2
                < self.radius_degrees**2
            )
            .select([ts_col, "rolling_lat", "rolling_lon"])
            .group_by(ts_col)
            .agg([
                pl.col("rolling_lat").alias("rolling_lat"),
                pl.col("rolling_lon").alias("rolling_lon"),
            ])
            .join(input, on=ts_col, how="left")
        )

        # sample equidistant points
        logger.debug("sampling equidistant points")
        df = (
            df.with_columns(
                pl.struct(["rolling_lon", "rolling_lat"])
                .map_elements(
                    self.sample_equidistant_points, return_dtype=pl.List(pl.Float64)
                )
                .alias("sampled_points")
            )
            .with_columns([
                pl.col("sampled_points")
                .list.slice(0, self.num_waypoints)
                .alias("waypoints_longitude"),
                pl.col("sampled_points")
                .list.slice(self.num_waypoints, self.num_waypoints)
                .alias("waypoints_latitude"),
            ])
            .drop("sampled_points", "rolling_lat", "rolling_lon")
        )

        heading_col = "Waypoints.heading_rad"
        df = self.heading_from_waypoints(
            df, x_col=lon_col, y_col=lat_col, out_col=heading_col
        )
        df = df.with_columns((np.pi / 2 - pl.col(heading_col)).alias(heading_col))

        df = self.build_heading_triangle(
            df,
            l=self._approximate_radius_deg(20),
            ego_lat_col=lat_col,
            ego_lon_col=lon_col,
            heading_col=heading_col,
        )

        # center waypoints
        # QUESTION: should we center to smoothed?
        df = df.with_columns(
            (pl.col("waypoints_latitude") - pl.col(lat_col)).alias("delta_lat"),
            (pl.col("waypoints_longitude") - pl.col(lon_col)).alias("delta_lon"),
        )

        # rotate to always point north
        logger.debug("rotating to always point north")
        df = df.with_columns(
            (
                pl.col("delta_lon") * pl.col(heading_col).cos()
                + pl.col("delta_lat") * pl.col(heading_col).sin().neg()
            ).alias("delta_lon_shifted"),
            (
                pl.col("delta_lat") * pl.col(heading_col).sin()
                + pl.col("delta_lon") * pl.col(heading_col).cos()
            ).alias("delta_lat_shifted"),
        ).drop("delta_lat", "delta_lon")

        logger.debug("converting deltas to list of tuples")
        df = df.with_columns(
            pl.struct(["delta_lon_shifted", "delta_lat_shifted"])
            .map_elements(
                lambda row: list(
                    map(
                        list,
                        zip(
                            row["delta_lon_shifted"],
                            row["delta_lat_shifted"],
                            strict=False,
                        ),
                    )
                )
            )
            .alias("Waypoints.delta")
        ).drop("delta_lat_shifted", "delta_lon_shifted")

        logger.debug("converting waypoints to list of tuples")
        df = df.with_columns(
            pl.struct(["waypoints_latitude", "waypoints_longitude"])
            .map_elements(
                lambda row: list(
                    zip(
                        row["waypoints_latitude"],
                        row["waypoints_longitude"],
                        strict=False,
                    )
                )
            )
            .alias("Waypoints.lat_lon")
        ).drop("waypoints_latitude", "waypoints_longitude")

        logger.debug("converting ego lat lon to list of tuples")
        # df = df.with_columns(
        #     pl.concat_list(["Gnss.latitude", "Gnss.longitude"]).alias(
        #         "Gnss.ego_lat_lon"
        #     )
        # ).drop("Gnss.latitude", "Gnss.longitude")

        logger.debug("waypoints sampled")
        return df

    def sample_equidistant_points(self, row: list[list[float]]) -> np.ndarray:
        """
        Sample a fixed number of equidistant points from a set of (x, y) coordinates.

        Parameters:
        points (np.ndarray): An array of shape (n, 2) containing (x, y) coordinates.
        num_samples (int): The desired number of equidistant points.

        Returns:
        np.ndarray: An array of shape (num_samples, 2) containing the sampled equidistant points.
        """
        points: np.ndarray[float] = np.column_stack((
            row["rolling_lon"],
            row["rolling_lat"],
        ))

        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        total_length = cumulative_distances[-1]
        sample_distances: np.ndarray[float] = np.linspace(
            0, total_length, self.num_waypoints
        )

        return np.column_stack((
            np.interp(sample_distances, cumulative_distances, points[:, 0]),
            np.interp(sample_distances, cumulative_distances, points[:, 1]),
        )).flatten(order="F")

    @staticmethod
    def heading_from_waypoints(
        df: pl.DataFrame,
        x_col: str = "Gnss.longitude",
        y_col: str = "Gnss.latitude",
        out_col: str = "heading_waypoints_rad",
        threshold: float = 0,
        window_size: int = 30,
    ) -> pl.DataFrame:
        """
        Calculate heading from waypoints by denosing x, y and then calculating angle betweeen next distinct points.

        Parameters:
        df (pl.DataFrame): Input dataframe.
        x_col (str): Name of the column with longitude.
        y_col (str): Name of the column with latitude.
        out_col (str): Name of the output column with heading.
        threshold (float): Threshold for denoising.
        window_size (int): Window size for denoising.

        Returns:
        pl.DataFrame: Dataframe with heading.
        """
        assert x_col in df.columns and y_col in df.columns, (
            f"Columns `{x_col}` and `{y_col}` not found in dataframe columns: {df.columns}"
        )

        # denoise x, y
        df = df.with_columns(
            pl.col(x_col)
            .rolling_mean(window_size=window_size, center=True)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .alias(f"{x_col}_ma"),
            pl.col(y_col)
            .rolling_mean(window_size=window_size, center=True)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .alias(f"{y_col}_ma"),
        )

        # angle by next distinct
        df = (
            df.with_columns(
                pl.col(f"{x_col}_ma").diff().alias("x_diff").shift(-1),
                pl.col(f"{y_col}_ma").diff().alias("y_diff").shift(-1),
            )
            .with_columns(
                pl.when(
                    (pl.col("x_diff").abs() <= threshold)
                    & (pl.col("y_diff").abs() <= threshold)
                )
                .then(None)
                .otherwise(pl.concat_list(["x_diff", "y_diff"]).list.to_array(width=2))
                .fill_null(strategy="backward")
                .alias("xy_diff")
            )
            .drop("x_diff", "y_diff")
        )

        # denoise
        df = (
            df.with_columns(
                pl.col("xy_diff")
                .arr.get(0)
                .rolling_mean(window_size=window_size, center=True)
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .alias("x_diff_ma"),
                pl.col("xy_diff")
                .arr.get(1)
                .rolling_mean(window_size=window_size, center=True)
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .alias("y_diff_ma"),
            )
            .with_columns(
                pl.arctan2(y=pl.col("y_diff_ma"), x=pl.col("x_diff_ma")).alias(out_col)
            )
            .drop("x_diff_ma", "y_diff_ma", "xy_diff")
        )

        return df

    @staticmethod
    def build_heading_triangle(
        df: pl.DataFrame, l: float, ego_lat_col: str, ego_lon_col: str, heading_col: str
    ) -> pl.DataFrame:
        """
        Build a triangle of points from the ego position and heading.
        """
        a_expr = [
            pl.col(ego_lat_col) + l * pl.col(heading_col).cos(),
            pl.col(ego_lon_col) + l * pl.col(heading_col).sin(),
        ]

        b_expr = [
            pl.col(ego_lat_col) - (l / 4) * pl.col(heading_col).sin(),
            pl.col(ego_lon_col) + (l / 4) * pl.col(heading_col).cos(),
        ]

        c_expr = [
            pl.col(ego_lat_col) + (l / 4) * pl.col(heading_col).sin(),
            pl.col(ego_lon_col) - (l / 4) * pl.col(heading_col).cos(),
        ]

        d_expr = [pl.col(ego_lat_col), pl.col(ego_lon_col)]

        return df.with_columns(
            pl.concat_list(
                a_expr[0],
                a_expr[1],
                b_expr[0],
                b_expr[1],
                c_expr[0],
                c_expr[1],
                d_expr[0],
                d_expr[1],
            ).alias("Waypoints.heading_triangle")
        )

    @staticmethod
    def _approximate_radius_deg(meters: float) -> float:
        lat = meters / 110574
        lon = meters / (111320 * np.cos(np.radians(lat)))
        return (lat + lon) / 2  # just a rough approximation since they are really close

    def _check_df(self, df: pl.DataFrame) -> None:
        # WARN: I guess it's no longer working

        if df["waypoints_longitude"].is_null().sum() > 0:
            logger.error(
                msg := "There are null values in the 'waypoints_longitude' column."
            )
            raise ValueError(msg)

        if df["waypoints_latitude"].is_null().sum() > 0:
            logger.error(
                msg := "There are null values in the 'waypoints_latitude' column."
            )
            raise ValueError(msg)

        if df[self.columns["time_stamp"]].unique().len() != df.shape[0]:
            logger.error(
                msg
                := "The number of unique timestamps in df_final does not match the number of rows."
            )
            raise ValueError(msg)

        if not df[self.columns["time_stamp"]].is_sorted():
            logger.error(msg := "The 'time_stamp' column in df_final is not sorted.")
            raise ValueError(msg)

        if not (
            df["waypoints_longitude"].list.len().min()
            == df["waypoints_longitude"].list.len().max()
            == self.num_waypoints
        ):
            logger.error(
                msg
                := "The lengths of lists in 'waypoints_longitude'  column are not equal to num_waypoints."
            )
            raise ValueError(msg)

        if not (
            df["waypoints_latitude"].list.len().min()
            == df["waypoints_latitude"].list.len().max()
            == self.num_waypoints
        ):
            logger.error(
                msg
                := "The lengths of lists in 'waypoints_latitude' column are not equal to num_waypoints."
            )
            raise ValueError(msg)
