"""Metadata updates and diagnostics must leave unrelated data untouched."""

from pathlib import Path
from stat import S_IMODE
from unittest.mock import patch

import pymupdf
import pytest

from artfinder.article_pdf import ArticlePDF


DOI = "10.1234/test"


def test_repair_rewrite_preserves_sibling_and_mode(tmp_path: Path) -> None:
    """Rewrite an actual repaired PDF without consuming an existing sibling."""
    path = tmp_path / "paper.pdf"
    sibling = tmp_path / "paper.pdf.tmp"
    sibling.write_bytes(b"unrelated")
    with pymupdf.open() as doc:
        doc.new_page().insert_text((70, 70), "Example.")
        raw = doc.tobytes()
    path.write_bytes(raw[:raw.rindex(b"\nxref\n")])
    path.chmod(0o600)
    with ArticlePDF(path) as article:
        assert not article.file.can_save_incrementally()
        article.set_doi(DOI)
        assert article.doi == DOI
    assert sibling.read_bytes() == b"unrelated"
    assert S_IMODE(path.stat().st_mode) == 0o600
    with ArticlePDF(path) as saved:
        assert saved.doi == DOI


@pytest.mark.parametrize("in_place", [False, True], ids=["copy", "rewrite"])
def test_full_doi_save_keeps_encryption(tmp_path: Path, in_place: bool) -> None:
    """Preserve the owner password and the original permission mask on full saves."""
    path = tmp_path / "restricted.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((70, 70), "Example.")
        doc.save(
            path, encryption=getattr(pymupdf, "PDF_ENCRYPT_AES_256"), owner_pw="owner",
            user_pw="", permissions=getattr(pymupdf, "PDF_PERM_PRINT"),
        )
    output = path if in_place else tmp_path / "copy.pdf"
    with ArticlePDF(path) as article:
        encryption = article.file.metadata["encryption"]
        permissions = article.file.permissions
        with patch.object(article.file, "can_save_incrementally", return_value=False):
            article.set_doi(DOI, output)
    with ArticlePDF(output) as saved:
        assert saved.doi == DOI
        assert saved.file.metadata["encryption"] == encryption
        assert saved.file.permissions == permissions
        assert saved.file.authenticate("owner") & 4


def test_mark_then_save_doi_does_not_mark_original(tmp_path: Path) -> None:
    """Only the diagnostic copy receives drawing commands, even after later saves."""
    path = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((70, 70), "Example.")
        doc.save(path)
    with ArticlePDF(path) as article:
        contents = article.file[0].read_contents()
        marked = article.mark("text", output_path=tmp_path)
        with pymupdf.open(marked) as copy:
            assert len(copy[0].get_drawings()) == 1
        assert article.file[0].read_contents() == contents
        article.set_doi(DOI)
    with pymupdf.open(path) as original:
        assert original[0].read_contents() == contents


@pytest.mark.parametrize("incremental", [True, False], ids=["append", "full-save"])
def test_mark_stages_live_encrypted_pdf_on_disk(tmp_path: Path, incremental: bool) -> None:
    """Mark the live state, not stale source bytes, without a PDF-sized bytes object."""
    path = tmp_path / "restricted.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((70, 70), "Example.")
        doc.save(path, encryption=getattr(pymupdf, "PDF_ENCRYPT_AES_256"), owner_pw="owner",
                 user_pw="", permissions=getattr(pymupdf, "PDF_PERM_PRINT"))
    original = path.read_bytes()
    output = tmp_path / "restricted_text.pdf"
    output.touch(mode=0o600)
    with ArticlePDF(path) as article:
        article.file.set_metadata({"title": "Unsaved edit"})
        with patch.object(article.file, "tobytes", side_effect=AssertionError("full memory copy")), patch.object(
            pymupdf.Document, "can_save_incrementally", return_value=incremental
        ):
            assert Path(article.mark("text", output_path=tmp_path)) == output
        with pymupdf.open(output) as marked:
            assert marked.metadata["title"] == "Unsaved edit"
            assert marked.metadata["encryption"] == article.file.metadata["encryption"]
            assert marked.permissions == article.file.permissions
            assert len(marked[0].get_drawings()) == 1
    assert path.read_bytes() == original
    assert S_IMODE(output.stat().st_mode) == 0o600


def test_failed_mark_preserves_previous_output(tmp_path: Path) -> None:
    """A failed mark must not publish a partial diagnostic or leave staging files."""
    with pymupdf.open() as doc:
        doc.new_page()
        source = doc.tobytes()
    output = tmp_path / "sample_text.pdf"
    output.write_bytes(b"previous result")
    with ArticlePDF(source, identifier="sample") as article, patch.object(
        article, "get_text_rects", side_effect=ValueError("cannot extract")
    ), pytest.raises(ValueError, match="cannot extract"):
        article.mark("text", output_path=tmp_path)
    assert output.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [output]


def test_failed_replace_leaves_original_readable(tmp_path: Path) -> None:
    """A failed replacement leaves the source on disk and its open reader usable."""
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.set_metadata({"subject": "doi:10.1234/original"})
        doc.save(path)
    original = path.read_bytes()
    with ArticlePDF(path) as article, patch.object(article.file, "can_save_incrementally", return_value=False):
        with patch.object(Path, "replace", side_effect=OSError("cannot replace")):
            with pytest.raises(OSError, match="cannot replace"):
                article.set_doi(DOI)
        assert article.doi == "10.1234/original"
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
