"""Guards for returning attachments as images for visual analysis.

Shared by Jira and Confluence: metadata is checked before download, so
unsupported or oversized files never cost a transfer, and the payload is
re-checked afterwards because metadata can be stale.
"""

from __future__ import annotations

from fastmcp.utilities.types import Image

from .render import format_size

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024

# The only mime types Claude can analyse visually, mapped to the format name
# FastMCP's Image helper expects.
IMAGE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def check_image_metadata(filename: str, mime_type: str, size: int) -> str:
    """Return the image format, or raise ValueError explaining the rejection."""
    image_format = IMAGE_FORMATS.get(mime_type.lower())
    if image_format is None:
        supported = ", ".join(sorted(set(IMAGE_FORMATS)))
        raise ValueError(
            f"Attachment '{filename}' has type '{mime_type or 'unknown'}', which "
            f"cannot be analysed visually. Only images are supported: {supported}. "
            "Videos and other file types cannot be fetched."
        )
    if size > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"Attachment '{filename}' is {format_size(size)}, above the "
            f"{format_size(MAX_ATTACHMENT_BYTES)} limit for inline analysis."
        )
    return image_format


def image_from_payload(filename: str, data: bytes, image_format: str) -> Image:
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"Attachment '{filename}' downloaded as {format_size(len(data))}, above "
            f"the {format_size(MAX_ATTACHMENT_BYTES)} limit for inline analysis."
        )
    if not data:
        raise ValueError(f"Attachment '{filename}' has no content.")
    return Image(data=data, format=image_format)
