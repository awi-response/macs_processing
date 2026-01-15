import argparse
import subprocess
from pathlib import Path

from macs_processing.utils.loading import (
    import_module_as_namespace,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--settings", type=Path, required=True, help="Path to Settings file"
)


args = parser.parse_args()

# import settings
settings = import_module_as_namespace(args.settings)


def main():
    """Run Pix4D processing on the generated project file."""
    # input dirs and names
    project_dir = Path(settings.PROJECT_DIR)
    project_name = settings.SITE_NAME
    output_file = project_dir / "04_pix4d" / f"{project_name}.p4d"

    pix4d_path = Path("C:/Program Files/Pix4Dmapper/pix4dmapper.bat")

    if not pix4d_path.exists():
        raise FileNotFoundError(f"Pix4D executable not found at: {pix4d_path}")

    print(f"Running Pix4D on {output_file.name}...")

    result = subprocess.run(
        [
            str(pix4d_path),
            "-r",
            str(output_file),
            "--close-gui",
        ],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pix4D processing failed with exit code {result.returncode}"
        )

    print("✓ Pix4D processing completed successfully")


if __name__ == "__main__":
    main()
