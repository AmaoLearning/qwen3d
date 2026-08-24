from datetime import datetime
import sys


def _to_size_tuple(image_size):
    """Normalize image size to (width, height)."""
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    return 0, 0


def create_image_info(image_id, file_name, image_size, *_, **__):
    """Create a minimal COCO image dict.

    The upstream training code expects only standard COCO image fields to be present.
    """
    width, height = _to_size_tuple(image_size)
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "license": 1,
        "date_captured": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


def create_annotation_info(
    annotation_id,
    image_id,
    category_id,
    binary_mask=None,
    area=None,
    is_crowd=0,
    bbox=None,
    segmentation=None,
):
    # This project only uses image infos from this helper module.
    raise NotImplementedError("annotations are not generated through this helper in this workflow")


pycococreatortools = sys.modules[__name__]
