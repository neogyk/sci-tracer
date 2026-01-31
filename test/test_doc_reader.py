from scitracer.tools.file_reader import doc_reader
import pytest

@pytest( "https://arxiv.org/pdf/2408.09869")
def test_doc_reader(url):
      md = doc_reader()

      assert len(md)>0
      assert md is not None