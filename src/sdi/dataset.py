"""NEU-DET steel surface images: download at a pinned revision, splits and labels.

The images come from the NEU Surface Defect Database (Song and Yan, 2013) in the cleaned release
``KeenForgeAI/NEU-DET-corrected``, which removes duplicates and adds a seeded, class-stratified
80/10/10 split. The upstream data has no stated licence, so this repository does not ship it;
:func:`download` fetches it and checks every file against its SHA-256.
"""

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

REPOSITORY = "KeenForgeAI/NEU-DET-corrected"
REVISION = "cbb799e5e5b8198d6228d94778f835082e42883c"
CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
SPLITS = ("train", "val", "test")

Split = Literal["train", "val", "test"]


class DatasetError(Exception):
    """The dataset is missing, incomplete or does not match its checksums."""


@dataclass(frozen=True, slots=True)
class Sample:
    """One image with its class index into :data:`CLASSES`."""

    path: Path
    label: int


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file of the dataset repository with its expected SHA-256."""

    path: str
    sha256: str


def label_of(name: str) -> int:
    """The class index of an image name such as ``rolled-in_scale_12``.

    Raises:
        DatasetError: If the name starts with no known class.
    """
    for index, cls in enumerate(CLASSES):
        number = name[len(cls) + 1 :].split(".", maxsplit=1)[0]
        if name.startswith(f"{cls}_") and number.isdigit():
            return index
    raise DatasetError(f"no class in image name {name!r}")


def load_split(root: Path, split: Split) -> list[Sample]:
    """The samples of one split, in the order of its list file.

    Raises:
        DatasetError: If the list file or an image is missing.
    """
    listing = root / "ImageSets" / f"{split}.txt"
    if not listing.is_file():
        raise DatasetError(f"{listing} is missing; run `sdi download` first")
    samples = []
    for name in listing.read_text(encoding="utf-8").split():
        path = root / "images" / split / f"{name}.jpg"
        if not path.is_file():
            raise DatasetError(f"{path} is missing; run `sdi download` again")
        samples.append(Sample(path, label_of(name)))
    return samples


def remote_files(client: httpx.Client) -> Iterator[RemoteFile]:
    """The images and split lists of the pinned revision, following the API's pages."""
    base = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}"
    for folder in ("ImageSets", *(f"images/{split}" for split in SPLITS)):
        url: str | None = f"{base}/{folder}"
        while url is not None:
            response = client.get(url)
            response.raise_for_status()
            for entry in response.json():
                if entry["type"] == "file":
                    sha = entry["lfs"]["oid"] if "lfs" in entry else None
                    yield RemoteFile(entry["path"], sha or "")
            url = response.links.get("next", {}).get("url")


def download(
    root: Path,
    client: httpx.Client | None = None,
    progress: Callable[[int], None] = lambda _: None,
) -> int:
    """Download what is missing or changed into ``root`` and check every image.

    Files without an LFS hash, the split lists, are taken as they are.

    Args:
        root: Target directory.
        client: HTTP client; a new one if unset.
        progress: Called with the number of files checked so far.

    Returns:
        The number of files downloaded.

    Raises:
        DatasetError: If a downloaded file does not match its SHA-256.
    """
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    fetched = 0
    try:
        for count, remote in enumerate(remote_files(client), start=1):
            target = root / remote.path
            if not (target.is_file() and _matches(target.read_bytes(), remote.sha256)):
                url = (
                    f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{remote.path}"
                )
                response = client.get(url)
                response.raise_for_status()
                if not _matches(response.content, remote.sha256):
                    raise DatasetError(f"{remote.path} does not match its SHA-256")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                fetched += 1
            progress(count)
    finally:
        if own:
            client.close()
    return fetched


def _matches(content: bytes, sha256: str) -> bool:
    return not sha256 or hashlib.sha256(content).hexdigest() == sha256
