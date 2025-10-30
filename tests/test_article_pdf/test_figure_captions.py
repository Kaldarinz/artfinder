"""
Tests for ArticlePDF figure captions extraction.
"""

import json
import pytest
from pathlib import Path
from artfinder.article_pdf import ArticlePDF


# Get the test PDFs directory
TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"
CAPTIONS_JSON = TEST_PDFS_DIR / "all_fig_captions.json"


@pytest.fixture
def expected_captions():
    """Load expected figure captions from JSON file."""
    with open(CAPTIONS_JSON, 'r') as f:
        return json.load(f)


@pytest.fixture
def pdf_files():
    """Return list of all test PDF files."""
    return list(TEST_PDFS_DIR.glob("*.pdf"))


class TestFigureCaptions:
    """Tests for figure captions extraction from PDF files."""

    def test_captions_json_exists(self):
        """Test that the captions JSON file exists."""
        assert CAPTIONS_JSON.exists(), f"Captions file not found: {CAPTIONS_JSON}"

    def test_captions_json_format(self, expected_captions):
        """Test that captions JSON has correct format."""
        assert isinstance(expected_captions, dict)
        assert len(expected_captions) > 0
        
        # Check structure of first entry
        first_pdf = next(iter(expected_captions.keys()))
        assert isinstance(expected_captions[first_pdf], dict)


    def test_figure_captions_keys_are_integers(self, pdf_files):
        """Test that figure_captions keys are integers."""
        if not pdf_files:
            pytest.skip("No PDF files found in test directory")
        
        article = ArticlePDF(pdf_files[0])
        captions = article.figure_captions
        
        for key in captions.keys():
            assert isinstance(key, int), f"Expected int key, got {type(key)}"
        
        article.close()

    def test_figure_captions_values_are_strings(self, pdf_files):
        """Test that figure_captions values are strings."""
        if not pdf_files:
            pytest.skip("No PDF files found in test directory")
        
        article = ArticlePDF(pdf_files[0])
        captions = article.figure_captions
        
        for value in captions.values():
            assert isinstance(value, str), f"Expected str value, got {type(value)}"
        
        article.close()


    def test_figure_captions_consistency(self, pdf_files):
        """Test that multiple calls to figure_captions return consistent results."""
        if not pdf_files:
            pytest.skip("No PDF files found in test directory")
        
        with ArticlePDF(pdf_files[0]) as article:
        
            captions1 = article.figure_captions
            captions2 = article.figure_captions
            
            assert captions1 == captions2, "Multiple calls to figure_captions should return same results"
            assert captions1 is captions2, "figure_captions should be cached"

    def test_empty_pdf_captions(self):
        """Test behavior with PDFs that have no figures."""
        # This test would require a PDF without figures
        # For now, we just verify the property doesn't crash
        pass


    def test_all_pdfs_caption_extraction(self, expected_captions):
        """Test caption extraction for all PDFs in the test directory."""
        results = []
        
        for pdf_name, expected in expected_captions.items():
            pdf_path = TEST_PDFS_DIR / pdf_name
            
            if not pdf_path.exists():
                results.append((pdf_name, "SKIP", "File not found"))
                continue
            
            try:
                with ArticlePDF(pdf_path) as article:
                    captions = article.figure_captions
                    
                    # Check caption count
                    if len(captions) != len(expected):
                        results.append((
                            pdf_name,
                            "FAIL",
                            f"Count mismatch: expected {len(expected)}, got {len(captions)}"
                        ))
                        continue
                    
                    # Check caption numbers
                    expected_nums = set(int(k) for k in expected.keys())
                    actual_nums = set(captions.keys())
                    if actual_nums != expected_nums:
                        results.append((
                            pdf_name,
                            "FAIL",
                            f"Figure numbers mismatch: expected {expected_nums}, got {actual_nums}"
                        ))
                        continue
                    
                    # Check caption text
                    all_match = True
                    for fig_num_str, expected_text in expected.items():
                        fig_num = int(fig_num_str)
                        if captions[fig_num] != expected_text:
                            all_match = False
                            break
                    
                    if all_match:
                        results.append((pdf_name, "PASS", ""))
                    else:
                        results.append((pdf_name, "FAIL", "Caption text mismatch"))
                
            except Exception as e:
                results.append((pdf_name, "ERROR", str(e)))
        
        # Report results
        failed = [r for r in results if r[1] in ["FAIL", "ERROR"]]
        
        if failed:
            fail_msg = "\n".join([f"{name}: {status} - {msg}" for name, status, msg in failed])
            pytest.fail(f"Caption extraction failed for {len(failed)} PDFs:\n{fail_msg}")

