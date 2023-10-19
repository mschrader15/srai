"""Test case generation for Hex2VecEmbedder."""
from pathlib import Path
from typing import Optional

import geopandas as gpd
import h3
from h3ronpy.pandas.vector import cells_to_polygons
from pytorch_lightning import seed_everything

from srai.constants import REGIONS_INDEX, WGS84_CRS
from srai.embedders.geovex.embedder import GeoVexEmbedder
from srai.h3 import ring_buffer_h3_regions_gdf
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER, OsmTagsFilter
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import geocode_to_region_gdf
from srai.regionalizers.h3_regionalizer import H3Regionalizer
from tests.embedders.geovex.constants import EMBEDDING_SIZE, TRAINER_KWARGS


def generate_test_case(
    test_case_name: str,
    geocoding_name: str,
    root_region_index: str,
    region_gen_radius: int,
    h3_res: int,
    model_radius: int,
    seed: int,
    tags: Optional[OsmTagsFilter] = None,
) -> None:
    """Generate test case for GeoVexEmbedder."""
    seed_everything(seed)

    if tags is None:
        tags = HEX2VEC_FILTER

    neighbourhood = H3Neighbourhood()
    regions_indexes = neighbourhood.get_neighbours_up_to_distance(
        root_region_index, region_gen_radius
    )
    regions_indexes.add(root_region_index)
    regions_indexes = list(regions_indexes)  # type: ignore

    geoms = cells_to_polygons([h3.str_to_int(r) for r in regions_indexes]).values
    regions_gdf = gpd.GeoDataFrame(index=regions_indexes, geometry=geoms, crs=WGS84_CRS)
    regions_gdf.index.name = REGIONS_INDEX

    area_gdf = geocode_to_region_gdf(geocoding_name)

    regionalizer = H3Regionalizer(resolution=h3_res)
    base_h3_regions = regionalizer.transform(area_gdf)

    regions_gdf = ring_buffer_h3_regions_gdf(base_h3_regions, distance=model_radius)
    buffered_geometry = regions_gdf.unary_union

    loader = OSMPbfLoader()
    features_gdf = loader.load(buffered_geometry, tags)

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions_gdf, features_gdf)

    neighbourhood = H3Neighbourhood(regions_gdf)

    embedder = GeoVexEmbedder(
        target_features=(
            [f"{super_}_{sub}" for super_, subs in tags.items() for sub in subs]  # type: ignore
        ),
        neighbourhood=neighbourhood,
        batch_size=10,
        neighbourhood_radius=model_radius,
        convolutional_layers=2,
        embedding_size=EMBEDDING_SIZE,
    )

    results_df = embedder.fit_transform(
        regions_gdf=regions_gdf,
        features_gdf=features_gdf,
        joint_gdf=joint_gdf,
        neighbourhood=neighbourhood,
        trainer_kwargs=TRAINER_KWARGS,
        learning_rate=0.001,
    )

    results_df.columns = results_df.columns.astype(str)

    files_prefix = f"{test_case_name}"

    output_path = Path(__file__).parent / "test_files"
    regions_gdf.to_parquet(output_path / f"{files_prefix}_regions.parquet")
    features_gdf.to_parquet(output_path / f"{files_prefix}_features.parquet")
    joint_gdf.to_parquet(output_path / f"{files_prefix}_joint.parquet")
    results_df.to_parquet(output_path / f"{files_prefix}_result.parquet")