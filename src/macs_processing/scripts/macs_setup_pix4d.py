import argparse

# ignore warnings
from pathlib import Path

import pandas as pd

from macs_processing.utils.loading import (
    import_module_as_namespace,
)
from macs_processing.utils.pix4d import (
    add_default_classification_objects,
    add_images_from_dataframe,
    add_processing_area_from_shapefile,
    export_p4d_to_xml,
    filter_images,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--settings", type=Path, required=True, help="Path to Settings file"
)


args = parser.parse_args()

# import settings
settings = import_module_as_namespace(args.settings)


def main():
    """Set up Pix4D project XML configuration."""
    # template file
    p4d_file = "pix4D_processing_templates/p4d_template.p4d"

    # input dirs and names
    project_dir = Path(settings.PROJECT_DIR)
    project_name = settings.SITE_NAME
    output_file = project_dir / "04_pix4d" / f"{project_name}.p4d"

    export_p4d_to_xml(p4d_file, output_file, image_resolution=0.5)

    df_nir, df_rgb = filter_images(project_dir)

    # nir
    df_nir["camera_name"] = "MACS_Polar1-2023_NIR99683_v2"
    df_nir["camera_id"] = 1
    df_nir["group"] = "NIR"

    # # rgb
    df_rgb["camera_name"] = "MACS_Polar1-2024_RGB111498_v1"
    df_rgb["camera_id"] = 0
    df_rgb["group"] = "RGB"

    df_merged = pd.concat([df_nir, df_rgb], ignore_index=True)

    add_images_from_dataframe(output_file, df_merged)

    add_processing_area_from_shapefile(
        output_file,
        project_dir / "02_studysites" / "AOI.shp",
        target_epsg=32608,  # UTM zone 8N (default)
    )

    add_default_classification_objects(output_file)


if __name__ == "__main__":
    main()
