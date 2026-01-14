import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
from rich.progress import track


def indent(elem, level=0):
    """Add pretty-printing indentation to XML elements."""
    indent_str = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str


def export_p4d_to_xml(p4d_file_path: Union[str, Path], output_xml_path: Union[str, Path], image_resolution: float) -> None:
    """
    Export Pix4D project file (.p4d) to XML with customizable image resolution.
    
    Args:
        p4d_file_path: Path to the input .p4d file
        output_xml_path: Path to the output XML file
        image_resolution: Image resolution value to set in the XML (e.g., 0.5, 1.0, 2.0)
    
    Raises:
        FileNotFoundError: If the input .p4d file does not exist
        Exception: If XML parsing or writing fails
    
    Example:
        export_p4d_to_xml(
            'WC_DempsterRTS02_20250803_15cm_98.p4d',
            'output.xml',
            image_resolution=0.5
        )
    """
    p4d_file_path = Path(p4d_file_path)
    output_xml_path = Path(output_xml_path)
    
    # Check if input file exists
    if not p4d_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {p4d_file_path}")
    
    # Parse the XML
    tree = ET.parse(p4d_file_path)
    root = tree.getroot()
    
    # Update the image resolution in the initial options
    initial_elem = root.find(".//options/initial")
    if initial_elem is not None:
        image_scale_elem = initial_elem.find("imageScale")
        if image_scale_elem is not None:
            image_scale_elem.text = str(image_resolution)
        else:
            # Create the element if it doesn't exist
            image_scale_elem = ET.Element("imageScale")
            image_scale_elem.text = str(image_resolution)
            initial_elem.append(image_scale_elem)
    
    # Remove contents of <inputs> while keeping <cameras>, <coordinateSystems>, <gcps>, <radiometry>
    inputs_elem = root.find(".//inputs")
    if inputs_elem is not None:
        # Elements to keep
        cameras_elem = inputs_elem.find("cameras")
        coord_sys_elem = inputs_elem.find("coordinateSystems")
        gcps_elem = inputs_elem.find("gcps")
        radiometry_elem = inputs_elem.find("radiometry")
        
        inputs_elem.clear()
        
        # Re-add elements in correct order
        if cameras_elem is not None:
            inputs_elem.append(cameras_elem)
        if coord_sys_elem is not None:
            inputs_elem.append(coord_sys_elem)
        if gcps_elem is not None:
            inputs_elem.append(gcps_elem)
        if radiometry_elem is not None:
            inputs_elem.append(radiometry_elem)
    
    # Remove contents of <images> tags
    images_elem = root.find(".//inputs/images")
    if images_elem is not None:
        images_elem.clear()
    
    # Format the XML with proper indentation
    indent(root)
    
    # Write to output XML file
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    
    print(f"✓ XML exported successfully to: {output_xml_path}")
    print(f"  Image resolution set to: {image_resolution}")


def add_image_to_xml(
    output_xml_path: Union[str, Path],
    filepath: str,
    group: str,
    camera_name: str,
    camera_id: int,
    altitude: float,
    latitude: float,
    longitude: float,
    omega: float,
    phi: float,
    kappa: float,
    enabled: bool = True,
    verbose: bool = False,
) -> None:
    """
    Add a single image to the output XML file.
    
    Args:
        output_xml_path: Path to the output XML file
        filepath: Image file path
        group: Image group (e.g., "RGB", "NIR")
        camera_name: Name of the camera
        camera_id: Camera ID
        altitude: Altitude in meters
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        omega: Omega angle (rotation)
        phi: Phi angle (rotation)
        kappa: Kappa angle (rotation)
        enabled: Whether the image is enabled (default: True)
    
    Example:
        add_image_to_xml(
            'output.xml',
            filepath='E:/path/to/image.tif',
            group='RGB',
            camera_name='MACS_Polar1-2024_RGB111498_v1',
            camera_id=0,
            altitude=1276.99,
            latitude=67.2598,
            longitude=-135.2202,
            omega=-0.533903487,
            phi=0.0786056368,
            kappa=100.105342564
        )
    """
    output_xml_path = Path(output_xml_path)
    
    if not output_xml_path.exists():
        raise FileNotFoundError(f"Output XML file not found: {output_xml_path}")
    
    # Parse the XML
    tree = ET.parse(output_xml_path)
    root = tree.getroot()
    
    # Find or create images element
    inputs_elem = root.find(".//inputs")
    if inputs_elem is None:
        raise ValueError("No <inputs> element found in XML")
    
    images_elem = inputs_elem.find("images")
    if images_elem is None:
        images_elem = ET.Element("images")
        inputs_elem.append(images_elem)
    
    # Create image element
    image_elem = ET.Element("image")
    image_elem.set("path", filepath)
    image_elem.set("group", group)
    image_elem.set("enabled", "true" if enabled else "false")
    
    # Add camera reference
    camera_elem = ET.SubElement(image_elem, "camera")
    camera_elem.set("name", camera_name)
    camera_elem.set("id", str(camera_id))
    
    # Add GPS data
    gps_elem = ET.SubElement(image_elem, "gps")
    gps_elem.set("alt", str(altitude))
    gps_elem.set("lat", str(latitude))
    gps_elem.set("lng", str(longitude))
    
    # Add XYZ (defaults to 0,0,0)
    xyz_elem = ET.SubElement(image_elem, "xyz")
    xyz_elem.set("x", "0")
    xyz_elem.set("y", "0")
    xyz_elem.set("z", "0")
    
    # Add accuracy XY
    accuracy_xy_elem = ET.SubElement(image_elem, "accuracyXY")
    accuracy_xy_elem.text = "0.05"
    
    # Add accuracy Z
    accuracy_z_elem = ET.SubElement(image_elem, "accuracyZ")
    accuracy_z_elem.text = "0.05"
    
    # Add OPK (omega, phi, kappa)
    opk_elem = ET.SubElement(image_elem, "opk")
    opk_elem.set("omega", str(omega))
    opk_elem.set("phi", str(phi))
    opk_elem.set("kappa", str(kappa))
    
    # Add accuracy OPK (defaults to 6, 4, 6)
    accuracy_opk_elem = ET.SubElement(image_elem, "accuracyOpk")
    accuracy_opk_elem.set("omega", "6")
    accuracy_opk_elem.set("phi", "4")
    accuracy_opk_elem.set("kappa", "6")
    
    # Add image to images element
    images_elem.append(image_elem)
    
    # Format the XML with proper indentation
    indent(root)
    
    # Write to output XML file
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    
    if verbose:
        print(f"✓ Image added: {filepath}")


def add_images_from_dataframe(output_xml_path: Union[str, Path], df: pd.DataFrame) -> None:
    """
    Add multiple images from a pandas DataFrame to the output XML file.
    
    Args:
        output_xml_path: Path to the output XML file
        df: Pandas DataFrame with columns: filepath, group, camera_name, camera_id, 
            altitude, latitude, longitude, omega, phi, kappa
    
    Example:
        import pandas as pd
        df = pd.read_csv('images.csv')
        add_images_from_dataframe('output.xml', df)
    """
    for idx, row in track(df.iterrows(), total=len(df), description="Adding images..."):
        add_image_to_xml(
            output_xml_path,
            filepath=row['file_path'].as_posix(),
            group=row['group'],
            camera_name=row['camera_name'],
            camera_id=int(row['camera_id']),
            altitude=float(row['Alt[m] ']),
            latitude=float(row['y']),
            longitude=float(row['x']),
            omega=float(row['Omega[deg] ']),
            phi=float(row['Phi[deg] ']),
            kappa=float(row['Kappa[deg]']),
            enabled=row.get('enabled', True),
            verbose=False
        )
    
    print(f"✓ Total images added: {len(df)}")


def add_processing_area_from_shapefile(output_xml_path: Union[str, Path], shapefile_path: Union[str, Path], target_epsg: int = 32608) -> None:
    """
    Add processing area to the output XML from a shapefile polygon.
    
    The polygon is transformed from lat/lon (EPSG:4326) to the target projection (default: UTM zone 8N, EPSG:32608),
    and the vertices are added as <geoCoord2D> elements.
    
    Args:
        output_xml_path: Path to the output XML file
        shapefile_path: Path to the shapefile containing the processing area polygon
        target_epsg: Target EPSG code for the projection (default: 32608 = UTM zone 8N)
    
    Example:
        add_processing_area_from_shapefile(
            'output.xml',
            'processing_area.shp',
            target_epsg=32608
        )
    """
    output_xml_path = Path(output_xml_path)
    shapefile_path = Path(shapefile_path)
    
    if not output_xml_path.exists():
        raise FileNotFoundError(f"Output XML file not found: {output_xml_path}")
    
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    if gdf.empty:
        raise ValueError("Shapefile is empty")
    
    # Get the first geometry
    geometry = gdf.geometry.iloc[0]
    
    # Ensure it's in lat/lon (EPSG:4326)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        geometry = gdf.geometry.iloc[0]
    
    # Transform to target projection
    geom_transformed = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(
        f"EPSG:{target_epsg}"
    )[0]
    
    # Extract coordinates from the exterior ring
    if geom_transformed.geom_type == "Polygon":
        coords = list(geom_transformed.exterior.coords)[
            :-1
        ]  # Remove duplicate last point
    elif geom_transformed.geom_type == "MultiPolygon":
        # Use the largest polygon
        largest_poly = max(geom_transformed.geoms, key=lambda p: p.area)
        coords = list(largest_poly.exterior.coords)[:-1]
    else:
        raise ValueError(f"Unsupported geometry type: {geom_transformed.geom_type}")
    
    # Parse the XML
    tree = ET.parse(output_xml_path)
    root = tree.getroot()
    
    # Find or create processingArea element
    inputs_elem = root.find(".//inputs")
    if inputs_elem is None:
        raise ValueError("No <inputs> element found in XML")
    
    # Remove existing processingArea if it exists
    existing_pa = inputs_elem.find("processingArea")
    if existing_pa is not None:
        inputs_elem.remove(existing_pa)
    
    # Create processingArea element
    processing_area_elem = ET.Element("processingArea")
    
    # Add coordinates as geoCoord2D elements
    for x, y in coords:
        coord_elem = ET.SubElement(processing_area_elem, "geoCoord2D")
        coord_elem.set("x", str(x))
        coord_elem.set("y", str(y))
    
    # Insert processingArea after images (find correct position)
    images_elem = inputs_elem.find("images")
    if images_elem is not None:
        images_index = list(inputs_elem).index(images_elem)
        inputs_elem.insert(images_index + 1, processing_area_elem)
    else:
        # If no images, append at the end
        inputs_elem.append(processing_area_elem)
    
    # Format the XML with proper indentation
    indent(root)
    
    # Write to output XML file
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    
    print(f"✓ Processing area added with {len(coords)} vertices from: {shapefile_path}")


def add_default_classification_objects(output_xml_path: Union[str, Path]) -> None:
    """
    Add default classification objects to the output XML at the end of <inputs> block.
    
    This adds:
    - <indexRegions/> (empty)
    - <indexClasses/> (empty)
    - <objects> with default point group classifications
    
    Args:
        output_xml_path: Path to the output XML file
    
    Example:
        add_default_classification_objects('output.xml')
    """
    output_xml_path = Path(output_xml_path)
    
    if not output_xml_path.exists():
        raise FileNotFoundError(f"Output XML file not found: {output_xml_path}")
    
    # Parse the XML
    tree = ET.parse(output_xml_path)
    root = tree.getroot()
    
    # Find inputs element
    inputs_elem = root.find(".//inputs")
    if inputs_elem is None:
        raise ValueError("No <inputs> element found in XML")
    
    # Remove existing elements if they exist
    for elem_name in ["indexRegions", "indexClasses", "objects"]:
        existing = inputs_elem.find(elem_name)
        if existing is not None:
            inputs_elem.remove(existing)
    
    # Create indexRegions (empty)
    index_regions_elem = ET.Element("indexRegions")
    inputs_elem.append(index_regions_elem)
    
    # Create indexClasses (empty)
    index_classes_elem = ET.Element("indexClasses")
    inputs_elem.append(index_classes_elem)
    
    # Create objects element with default classifications
    objects_elem = ET.Element("objects")
    
    # Define default objects
    default_objects = [
        {"name": "Clipping Box", "type": "CLIPPING_BOX", "centerX": "490195", "centerY": "7459765", "centerZ": "1282", "sizeX": "5", "sizeY": "5", "sizeZ": "5", "rotationZ": "0", "fresh": "true"},
        {"name": "Unclassified", "type": "POINT_GROUP", "groupId": "0", "classificationId": "0", "locked": "false"},
        {"name": "Disabled", "type": "POINT_GROUP", "groupId": "1", "classificationId": "0", "locked": "false"},
        {"name": "Ground", "type": "POINT_GROUP", "groupId": "100", "classificationId": "0", "locked": "false"},
        {"name": "Road Surface", "type": "POINT_GROUP", "groupId": "106", "classificationId": "0", "locked": "false"},
        {"name": "High Vegetation", "type": "POINT_GROUP", "groupId": "102", "classificationId": "0", "locked": "false"},
        {"name": "Building", "type": "POINT_GROUP", "groupId": "103", "classificationId": "0", "locked": "false"},
        {"name": "Human Made Object", "type": "POINT_GROUP", "groupId": "109", "classificationId": "0", "locked": "false"},
    ]
    
    # Add objects
    for obj_data in default_objects:
        obj_elem = ET.SubElement(objects_elem, "object")
        for key, value in obj_data.items():
            obj_elem.set(key, value)
    
    inputs_elem.append(objects_elem)
    
    # Format the XML with proper indentation
    indent(root)
    
    # Write to output XML file
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    
    print(f"✓ Default classification objects added to: {output_xml_path}")


def filter_images(base_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter images by type (NIR and RGB) and match them with navigation data.
    
    Args:
        base_dir: Base directory containing 01_rawdata/tif subdirectories
    
    Returns:
        tuple: (df_nir, df_rgb) - DataFrames with NIR and RGB images respectively
    
    Example:
        df_nir, df_rgb = filter_images('E:/MACS/2025/WC_DempsterRTS02_20250803_15cm_98')
    """
    base_dir = Path(base_dir)
    image_dir = base_dir / "01_rawdata" / "tif"
    nir_dir = image_dir / "99683_NIR"
    rgb_dir = image_dir / "111498_RGB"
    navfile = image_dir / "geo_pix4d_new.txt"

    nir_files = list(nir_dir.glob("*.tif"))
    rgb_files = list(rgb_dir.glob("*.tif"))

    df = pd.read_table(navfile, delimiter="\t")
    df_nir = df[df["imagename_tif"].isin([nir.name for nir in nir_files])]
    df_rgb = df[df["imagename_tif"].isin([rgb.name for rgb in rgb_files])]

    df_rgb["file_path"] = df_rgb.apply(lambda x: rgb_dir / x["imagename_tif"], axis=1)
    df_nir["file_path"] = df_nir.apply(lambda x: nir_dir / x["imagename_tif"], axis=1)

    return df_nir, df_rgb
