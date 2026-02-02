### Run and evaluate the performance of the AI-Scientist using different problems:
### 1. SokanaAI Scientist: https://sakana.ai/ai-scientist/
import sys

import dspy
import asyncio
from httpx_aiohttp import HttpxAiohttpClient
import pdb
import tqdm
from docling.document_converter import DocumentConverter

def doc_reader(source:str):
      converter = DocumentConverter()
      result = converter.convert(source)
      return result.document.export_to_markdown()


# def fetch():
#       async with HttpxAiohttpClient() as client:
#             response = await client.get("https://encode.io")
#             print(response)

#       asyncio.run(main())

class Trajectory(dspy.Signature):
      paragraph = dspy.InputField(desc="Part of the reseach article")
      problem:str = dspy.OutputField(desc='the formulation of problem that is being solve in the paragraph or article. One sentence. ')
      method:str = dspy.OutputField(desc="short list of the insturction that necessary to proceed to get the result. Several Sentences")
      result:str = dspy.OutputField(desc="The result or conclusion of the paragraph. Provide one sentence")

class SearchAgent(dspy.Module):
      """
      The search agent uses the available tools for the searching for the information 
      by given prompt
      """
      def __init__(self):
            self.search = dspy.Predict()
            self.search_tools = []
            self.fetch_tools = []
            return
      
      def fetch(self):
            return

      def forward(self, prompt:str):
            for tool in self.tools:
                  prediction = self.search(prompt)
                  result = self.fetch(prediction)
            return result
      
class RAG(dspy.Module):
      """
      The RAG Module provides the discovery trajectories for given idea, question or hypothesis.
      Please chunk the text of the document and decompose it into the sequence of the
      problem - method of the problem - solution and cosequences. This discovery trajectory for each of the paper
      will be used for the research planning.
      """
      def __init__(self):
            self.trajectories = dspy.Predict(Trajectory)
            self.tool = doc_reader
            return
      
      def build_rag(self, query:str):
            url = "https://arxiv.org/pdf/2412.09871"
            doc = self.tool(url)
            return doc
      
      def forward(self, prompt:str):
            tr_list = []
            paper = self.build_rag(query=prompt)
            for section in tqdm.tqdm(paper.split("##")):
                  if len(section)==0:
                        continue
                  try:
                        tr_list.append(self.trajectories(paragraph=section))
                  except Exception:
                        pass
            return tr_list
      
if __name__ == "__main__":

      import os
      os.environ["MISTRAL_API_KEY"] = "sBEjabQFOosPqTWOrBkJKKXFQK9RHCha"
      lm = dspy.LM("mistral/ministral-8b-2512")
      dspy.configure(lm=lm)
      idea = "How to create the byte free input layer for neural network "
      rag_agent = RAG()
      #Search agent:
      traj = rag_agent.forward(idea)
      dict_ = [{'p':t.problem, 's': t.method, 'c':t.result} for t in traj]
      with open("traj.json", "w") as f: json.dump(dict_, f)
      
      print(traj)
