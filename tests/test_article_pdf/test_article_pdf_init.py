"""
Tests for ArticlePDF class initialization.
"""

import pytest
from pathlib import Path
import tempfile
from artfinder.article_pdf import ArticlePDF


# Get the test PDFs directory
TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"


@pytest.fixture
def sample_pdf():
    """Return path to a sample PDF file."""
    pdf_files = list(TEST_PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in test directory")
    return pdf_files[0]


@pytest.fixture
def all_test_pdfs():
    """Return list of all test PDF files."""
    pdf_files = list(TEST_PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in test directory")
    return pdf_files


class TestArticlePDFInitialization:
    """Tests for ArticlePDF class initialization."""

    def test_init_with_valid_pdf_path_string(self, sample_pdf):
        """Test initialization with a valid PDF path as string."""
        article = ArticlePDF(str(sample_pdf))
        
        assert article is not None
        assert article.path == Path(sample_pdf)
        assert article.file is not None
        assert len(article.file) > 0  # Should have at least one page

    def test_init_with_valid_pdf_path_object(self, sample_pdf):
        """Test initialization with a valid PDF path as Path object."""
        article = ArticlePDF(Path(sample_pdf))
        
        assert article is not None
        assert article.path == Path(sample_pdf)
        assert article.file is not None

    def test_init_with_identifier(self, sample_pdf):
        """Test initialization with custom identifier."""
        custom_id = "test_article_123"
        article = ArticlePDF(sample_pdf, identifier=custom_id)
        
        assert article.identifier == custom_id

    def test_init_without_identifier(self, sample_pdf):
        """Test initialization without custom identifier uses filename."""
        article = ArticlePDF(sample_pdf)
        
        # Should use filename as identifier
        assert article.identifier == sample_pdf.name

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent PDF file raises FileNotFoundError."""
        nonexistent_path = "/path/to/nonexistent/file.pdf"
        
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            ArticlePDF(nonexistent_path)

    def test_init_with_invalid_pdf_file(self):
        """Test initialization with invalid PDF file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"This is not a valid PDF file")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="Failed to open PDF file"):
                ArticlePDF(tmp_path)
        finally:
            Path(tmp_path).unlink()


    def test_context_manager(self, sample_pdf):
        """Test ArticlePDF as context manager."""
        with ArticlePDF(sample_pdf) as article:
            assert article is not None
            assert article.file is not None
        
        # After exiting context, file should be closed
        # Note: pymupdf doesn't have a clear is_closed property, but we can verify
        # that it doesn't raise an exception

    def test_close_method(self, sample_pdf):
        """Test close method."""
        article = ArticlePDF(sample_pdf)
        assert article.file is not None
        
        article.close()
        # Verify close doesn't raise exception

    def test_multiple_pdfs_initialization(self, all_test_pdfs):
        """Test initialization with multiple PDF files."""
        articles = []
        
        for pdf_path in all_test_pdfs[:5]:  # Test first 5 PDFs
            article = ArticlePDF(pdf_path)
            assert article is not None
            assert article.file is not None
            articles.append(article)
        
        # Clean up
        for article in articles:
            article.close()

    def test_init_with_empty_identifier(self, sample_pdf):
        """Test initialization with empty string identifier uses filename."""
        article = ArticlePDF(sample_pdf, identifier="")
        
        assert article.identifier == sample_pdf.name

    def test_pdf_path_attribute(self, sample_pdf):
        """Test that path attribute is correctly set."""
        article = ArticlePDF(sample_pdf)
        
        assert isinstance(article.path, Path)
        assert article.path.exists()
        assert article.path.suffix == ".pdf"

