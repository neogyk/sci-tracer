import os
import fitz
from typing import List


class PDF_Reader:
    def __init__(self, preprocess=None):
        self.preprocess = preprocess
        return

    def read_file(self, path: str) -> List[str]:
        if not os.path.exists(path):
            raise Exception
        with fitz.open(path) as f:
            text = [page.get_text() for page in f]
        if self.preprocess is not None:
            text = [self.preprocess(t) for t in text]
        return text

    def _run_ocr(self, path: str) -> None:
        return
