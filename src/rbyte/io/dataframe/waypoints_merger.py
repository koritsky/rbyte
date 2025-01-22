from typing import final
import os
import json

import math
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@final
class DataFrameWaypointsMerger:
    __name__ = __qualname__
    
    ego_lat_col = 'Gnss.latitude'
    ego_lon_col = 'Gnss.longitude'
    wpts_lat_col = 'Waypoints.lat'
    wpts_lon_col = 'Waypoints.lon'
    wpts_lat_lon_col = 'Waypoints.lat_lon'
    heading_col = 'heading'

    def __init__(self, waypoints_path: os.PathLike, timestamp_column: str, num_waypoints: int = 10) -> None:
        self.waypoints_path = waypoints_path
        self.timestamp_column = timestamp_column
        self.num_waypoints = num_waypoints

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        df_wpts = self._build_df_wpts(self.waypoints_path)

        
        df = input.join_asof(
            df_wpts.select(['datetime', self.wpts_lat_col, self.wpts_lon_col, 'heading']),
            left_on=self.timestamp_column,
            right_on='datetime',
            strategy='nearest'
        ).drop('datetime')

        df = self._center_and_rotate(df)
        
        df = df.with_columns(
            pl.struct([self.wpts_lat_col, self.wpts_lon_col])
            .map_elements(
                lambda row: list(zip(row[self.wpts_lat_col], row[self.wpts_lon_col], strict=False))
            )
            .alias(self.wpts_lat_lon_col)
        ).drop(self.wpts_lat_col, self.wpts_lon_col)
        
        df = self.build_heading_triangle(
            df,
            l=self._approximate_radius_deg(20),
            ego_lat_col=self.ego_lat_col,
            ego_lon_col=self.ego_lon_col,
            heading_col=self.heading_col,
        )
        logger.debug("Waypoints merged")
        return df
    
    def _build_df_wpts(self, waypoints_path: os.PathLike) -> pl.DataFrame:
        waypoints_json = self._read_json(waypoints_path)
        df_wpts = self._json_to_df(waypoints_json)
        df_wpts = df_wpts.with_columns(datetime=pl.from_epoch(df_wpts['timestamp'] * 1e9, time_unit='ns' ))
        df_wpts = pl.concat([df_wpts] + [df_wpts[-1, :]] * (self.num_waypoints - 1))
        df_wpts = df_wpts.with_row_index().with_columns(index=pl.col('index').cast(pl.Int32))
        for col in ['lon', 'lat']:
            df_wpts = (
                df_wpts
                .group_by_dynamic(index_column='index', period=f'{self.num_waypoints}i', every='1i')
                .agg(pl.col(col).alias(f'Waypoints.{col}'))
                .join(df_wpts, on='index', how='left')
            )
        return df_wpts[:-(self.num_waypoints - 1)]

    def _json_to_df(self, json_data: dict) -> pl.DataFrame:
        data = {
            "timestamp": [],
            "heading": [],
            "lon": [],
            "lat": []
        }
        
        for feature in json_data["features"]:
            data["timestamp"].append(feature["properties"]["timestamp"])
            data["heading"].append(math.radians(feature["properties"]["heading"]))
            data["lon"].append(feature["geometry"]["coordinates"][0])
            data["lat"].append(feature["geometry"]["coordinates"][1])
        
        df_wpts = pl.DataFrame(data)
        return df_wpts


    def _read_json(self, file_path: os.PathLike) -> dict:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def _center_and_rotate(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.with_columns(  # center
                (pl.col(self.wpts_lat_col) - pl.col(self.ego_lat_col)).alias("wpts_lat_centered"),
                (pl.col(self.wpts_lon_col) - pl.col(self.ego_lon_col)).alias("wpts_lon_centered"),
            )
            .with_columns(  # rotate
                (
                    pl.col("wpts_lon_centered") * pl.col(self.heading_col).cos()
                    + pl.col("wpts_lat_centered") * pl.col(self.heading_col).sin().neg()
                ).alias("wpts_lon_centered_rotated"),
                (
                    pl.col("wpts_lon_centered") * pl.col(self.heading_col).sin()
                    + pl.col("wpts_lat_centered") * pl.col(self.heading_col).cos()
                ).alias("wpts_lat_centered_rotated"),
            )
            .drop("wpts_lat_centered", "wpts_lon_centered")
        )

        df = df.with_columns(
            pl.struct(["wpts_lon_centered_rotated", "wpts_lat_centered_rotated"])
            .map_elements(
                lambda row: list(
                    map(
                        list,
                        zip(
                            row["wpts_lon_centered_rotated"],
                            row["wpts_lat_centered_rotated"],
                            strict=False,
                        ),
                    )
                )
            )
            .alias("Waypoints.lon_lat_normalized")
        ).drop("wpts_lat_centered_rotated", "wpts_lon_centered_rotated")

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
        lon = meters / (111320 * math.cos(math.radians(lat)))
        return (lat + lon) / 2  # just a rough approximation since they are really close


