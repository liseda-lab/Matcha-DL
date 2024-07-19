from pathlib import Path

import yaml
import tarfile
import os
import urllib.request

__matcha_url__ = "https://github.com/liseda-lab/Matcha-DL/releases/download/JARv0.1.0/matcha_jar.tar.gz"
__matcha_dl_dir__ = Path(__file__).parent

# If matchaJar and dependencies don't exist, download them.

def download_macha():
    url = __matcha_url__
    download_path = __matcha_dl_dir__ / "impl/matcha/"
    filename = download_path / "macha.tar.gz"

    # Download the matcha directory
    print("Downloading Matcha-DL jar and dependencies...")
    urllib.request.urlretrieve(url, str(filename))

    # Uncompress the tar.gz file
    print("Uncompressing Matcha-DL jar and dependencies...")
    with tarfile.open(str(filename), 'r:gz') as tar_ref:
        tar_ref.extractall(download_path)

    # Remove the tar.gz file
    os.remove(str(filename))

# Check if the matcha directory exists
if not (__matcha_dl_dir__ / "impl/matcha/matcha/").exists():
    print("Matcha-DL jar and dependencies not found. Downloading...")
    download_macha()

# Get AlignmentRunner

from .delivery.api import AlignmentRunner
