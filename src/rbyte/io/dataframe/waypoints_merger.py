from typing import final
import os
import json

from math import sin, cos, radians
import polars as pl
from structlog import get_logger
import shapely
import geopandas as gpd
from pyproj import Transformer


logger = get_logger(__name__)


@final
class DataFrameWaypointsMerger:
    __name__ = __qualname__

    ego_lat_col = "Gnss.latitude"
    ego_lon_col = "Gnss.longitude"
    wpts_lat_col = "Waypoints.lat"
    wpts_lon_col = "Waypoints.lon"
    wpts_lat_lon_col = "Waypoints.gnss.lat_lon"
    heading_col = "heading"

    def __init__(
        self,
        waypoints_path: os.PathLike,
        timestamp_column: str,
        num_waypoints: int = 10,
        out_col: str = "Waypoints.xy",
        predict_mode: bool = False,
    ) -> None:
        self.waypoints_path = waypoints_path
        self.timestamp_column = timestamp_column
        self.num_waypoints = num_waypoints
        self.out_col = out_col
        self.predict_mode = predict_mode
        self.transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        logger.debug("Building waypoints dataframe")
        df_wpts = self._build_df_wpts(self.waypoints_path)

        logger.debug("Joining waypoints dataframe")
        df = input.join_asof(
            df_wpts.select([
                "datetime",
                self.wpts_lat_col,
                self.wpts_lon_col,
                self.heading_col,
            ]),
            left_on=self.timestamp_column,
            right_on="datetime",
            strategy="nearest",
        ).drop("datetime")

        logger.debug("Centering and rotating waypoints dataframe")
        df = self._center_and_rotate(df, out_col=self.out_col)

        if self.predict_mode:
            logger.debug("Building auxilary columns")
            df = df.with_columns(
                pl.struct([self.wpts_lat_col, self.wpts_lon_col])
                .map_elements(
                    lambda row: list(
                        zip(row[self.wpts_lat_col], row[self.wpts_lon_col], strict=False)
                    ),
                    return_dtype=pl.List(pl.List(pl.Float64)),
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
        return df[:-1] # hack to handle last frame rbyte problem

    def _build_df_wpts(self, waypoints_path: os.PathLike) -> pl.DataFrame:
        waypoints_json = self._read_json(waypoints_path)
        df_wpts = self._json_to_df(waypoints_json)
        df_wpts = df_wpts.with_columns(
            datetime=pl.from_epoch(df_wpts["timestamp"] * 1e9, time_unit="ns")
        ).sort("datetime")
        df_wpts = pl.concat([df_wpts] + [df_wpts[-1, :]] * (self.num_waypoints - 1)) # duplicate last waypoints
        df_wpts = df_wpts.with_row_index().with_columns(
            index=pl.col("index").cast(pl.Int32)
        )
        for col in ["lon", "lat"]:
            df_wpts = (
                df_wpts.group_by_dynamic(
                    index_column="index", period=f"{self.num_waypoints}i", every="1i"
                )
                .agg(pl.col(col).alias(f"Waypoints.{col}"))
                .join(df_wpts, on="index", how="left")
            )
        return df_wpts[: -(self.num_waypoints - 1)]

    def _json_to_df(self, json_data: dict) -> pl.DataFrame:
        data = {"timestamp": [], "heading": [], "lon": [], "lat": []}

        for feature in json_data["features"]:
            data["timestamp"].append(feature["properties"]["timestamp"])
            data["heading"].append(radians(feature["properties"]["heading"]))
            data["lon"].append(feature["geometry"]["coordinates"][0])
            data["lat"].append(feature["geometry"]["coordinates"][1])

        df_wpts = pl.DataFrame(data)
        return df_wpts

    def _read_json(self, file_path: os.PathLike) -> dict:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    def _center_and_rotate(
        self,
        df: pl.DataFrame, 
        out_col: str = 'Waypoints.xy', 
    ):
        """
        Usage of geopandas and shapely significantly improves performance.
        This function explodes lists of waypoints into one series of points and duplicate ego points accordingly.
        Then it calculates distance between each waypoint and ego position, rotates it and aggregate back into lists
        """

        # wpts
        x_wpts, y_wpts = self.transformer_to_utm.transform(
            df[self.wpts_lon_col].explode(), 
            df[self.wpts_lat_col].explode()
        )
        wpts_series = gpd.GeoSeries(shapely.points(x_wpts, y_wpts))

        # ego
        x_ego, y_ego = self.transformer_to_utm.transform(
            df.select(pl.col(self.ego_lon_col).repeat_by(self.num_waypoints).flatten().explode()), 
            df.select(pl.col(self.ego_lat_col).repeat_by(self.num_waypoints).flatten().explode())
        )
        ego_pos_series = gpd.GeoSeries(shapely.points(x_ego, y_ego)[:, 0])


        x_diff = wpts_series.x - ego_pos_series.x
        y_diff = wpts_series.y - ego_pos_series.y
        headings = df.select(pl.col(self.heading_col).repeat_by(self.num_waypoints).flatten().explode())

        df_ = (
            pl.DataFrame({'_x_diff': x_diff, '_y_diff': y_diff, "_heading": headings})
            .with_columns(
                (pl.col('_x_diff') * pl.col('_heading').cos() - pl.col('_y_diff') * pl.col('_heading').sin()).alias('x_diff'),
                (pl.col('_x_diff') * pl.col('_heading').sin() + pl.col('_y_diff') * pl.col('_heading').cos()).alias('y_diff')
            )
            .with_row_index()
            .with_columns(
                index=pl.col('index').cast(pl.Int32),
                xy=pl.concat_list([pl.col('x_diff'), pl.col('y_diff')])
            )
            .drop(['_x_diff', '_y_diff', '_heading'])
            .group_by_dynamic(
                index_column='index', period=f'{self.num_waypoints}i', every=f'{self.num_waypoints}i'
            ).agg(
                pl.col('xy').list.to_array(width=2).alias(out_col)
            )
            .drop('index')
        )
        return df.hstack(df_)

    # def _center_and_rotate(self, df: pl.DataFrame, out_col: str) -> pl.DataFrame:
    #     breakpoint()
    #     df = df.with_columns(
    #         pl.struct([self.ego_lat_col, self.ego_lon_col, self.wpts_lat_col, self.wpts_lon_col, self.heading_col])
    #         .map_elements(
    #             lambda row: self._delta_xy(
    #                 row[self.ego_lat_col],
    #                 row[self.ego_lon_col],
    #                 row[self.wpts_lat_col],
    #                 row[self.wpts_lon_col],
    #                 row[self.heading_col]
    #             ),
    #             return_dtype=pl.List(pl.List(pl.Float64)),
    #         )
    #         .alias(out_col)
    #     )
    #     return df
    
    # @staticmethod
    # def _delta_xy(p1_lat:float, p1_lon:float, p2_lat_list:list[float], p2_lon_list:list[float], heading:float):
    #     breakpoint()
    #     def calculate_delta(p2_lat, p2_lon):
    #         x_delta = geodesic((p1_lat, p1_lon), (p1_lat, p2_lon)).meters
    #         if p2_lon < p1_lon:
    #             x_delta = -x_delta
    #         y_delta = geodesic((p1_lat, p1_lon), (p2_lat, p1_lon)).meters
    #         if p2_lat < p1_lat:
    #             y_delta = -y_delta
    #         return (
    #             x_delta * cos(heading) - y_delta * sin(heading),
    #             x_delta * sin(heading) + y_delta * cos(heading)
    #         )

    #     delta_list = list(map(calculate_delta, p2_lat_list, p2_lon_list))
    #     return delta_list
    

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
            ).alias("Waypoints.gnss.heading_triangle")
        )

    @staticmethod
    def _approximate_radius_deg(meters: float) -> float:
        lat = meters / 110574
        lon = meters / (111320 * cos(radians(lat)))
        return (lat + lon) / 2  # just a rough approximation since they are really close
