import os
import xml.etree.ElementTree as ET
from tempfile import NamedTemporaryFile

ENV_ASSET_DIR_V1 = os.path.join(os.path.dirname(__file__), 'assets_v1')
ENV_ASSET_DIR_V2 = os.path.join(os.path.dirname(__file__), 'assets_v2')


def full_v1_path_for(file_name):
    return os.path.join(ENV_ASSET_DIR_V1, file_name)


def full_v2_path_for(file_name, transparent_sawyer=False):
    path = os.path.join(ENV_ASSET_DIR_V2, file_name)
    if not transparent_sawyer:
        return path
    fold, file_path = os.path.split(path)
    file_path = f"{file_path[:-len('.xml')]}_transparent_sawyer.xml"
    new_path = os.path.join(fold, file_path)
    tree = ET.parse(path)
    tree.getroot() \
        .find('worldbody').find('include') \
        .set('file', '../objects/assets/xyz_base_transparent.xml')
    tree.write(new_path)
    return new_path
