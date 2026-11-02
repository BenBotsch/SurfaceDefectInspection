import hashlib
import json
from pathlib import Path

import httpx
import pytest

from sdi.dataset import CLASSES, DatasetError, download, label_of, load_split

JPEG = b"\xff\xd8 not really a jpeg"


def test_label_comes_from_the_class_prefix() -> None:
    assert label_of("crazing_1") == 0
    assert label_of("rolled-in_scale_12") == CLASSES.index("rolled-in_scale")
    assert label_of("pitted_surface_7.jpg") == CLASSES.index("pitted_surface")


@pytest.mark.parametrize("name", ["rust_3", "crazing", "scratches_x"])
def test_unknown_names_are_an_error(name: str) -> None:
    with pytest.raises(DatasetError):
        label_of(name)


def test_split_lists_images_in_file_order(tmp_path: Path) -> None:
    (tmp_path / "ImageSets").mkdir()
    (tmp_path / "ImageSets" / "val.txt").write_text("scratches_2\ncrazing_1\n")
    (tmp_path / "images" / "val").mkdir(parents=True)
    for name in ("scratches_2", "crazing_1"):
        (tmp_path / "images" / "val" / f"{name}.jpg").write_bytes(JPEG)

    samples = load_split(tmp_path, "val")

    assert [(s.path.name, s.label) for s in samples] == [
        ("scratches_2.jpg", 5),
        ("crazing_1.jpg", 0),
    ]


def test_missing_split_says_to_download(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="sdi download"):
        load_split(tmp_path, "train")


def fake_hub(content: dict[str, bytes], sha: dict[str, str] | None = None) -> httpx.Client:
    """A Hub that lists ``content`` two entries per page and serves the files."""
    sha = sha or {
        p: hashlib.sha256(c).hexdigest() for p, c in content.items() if p.endswith(".jpg")
    }

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/api/datasets/" in path:
            folder = path.split("/tree/")[1].split("/", 1)[1]
            entries = [
                {"type": "file", "path": p} | ({"lfs": {"oid": sha[p]}} if p in sha else {})
                for p in content
                if p.rsplit("/", 1)[0] == folder
            ]
            page = int(request.url.params.get("cursor", "0"))
            headers = {}
            if page + 2 < len(entries):
                headers["Link"] = (
                    f'<{request.url.copy_merge_params({"cursor": page + 2})}>; rel="next"'
                )
            return httpx.Response(
                200, content=json.dumps(entries[page : page + 2]), headers=headers
            )
        return httpx.Response(200, content=content[path.split("/", 6)[6]])

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_download_follows_pages_and_skips_files_that_match(tmp_path: Path) -> None:
    content = {
        "ImageSets/train.txt": b"crazing_1\ncrazing_2\ncrazing_3\n",
        "images/train/crazing_1.jpg": b"1",
        "images/train/crazing_2.jpg": b"2",
        "images/train/crazing_3.jpg": b"3",
    }

    assert download(tmp_path, fake_hub(content)) == 4
    assert (tmp_path / "images/train/crazing_3.jpg").read_bytes() == b"3"
    assert len(load_split(tmp_path, "train")) == 3
    assert download(tmp_path, fake_hub(content)) == 0


def test_download_refuses_a_file_with_the_wrong_hash(tmp_path: Path) -> None:
    content = {"images/train/crazing_1.jpg": b"tampered"}

    with pytest.raises(DatasetError, match="SHA-256"):
        download(tmp_path, fake_hub(content, {"images/train/crazing_1.jpg": "0" * 64}))
    assert not (tmp_path / "images/train/crazing_1.jpg").exists()
