from rich import print
from rich.markdown import Markdown
from pdf_reader import PDF_Reader
from sci_tracer import SciTracer
from dataclasses import dataclass

# import argparse


@dataclass
class Config:
    prompt = "./prompt_template.json"


config = Config
file_path = "./examples/PMIHES_1988__68__175_0.pdf"
# read the file->
sci_tracer = SciTracer(config=config)
reader = PDF_Reader()
texts = reader.read_file(file_path)
results = sci_tracer(text=texts)
for res in results:
    print(Markdown(res.choices[0].message.content))
# add the prompt->
# Construct the Search in the PDF File:
# Linear, TreeSearch,
# Load the system Prompt:
# for each paper  construct the bidirectional tree of problems -> solution -> problem
