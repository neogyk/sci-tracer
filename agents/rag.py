### Run and evaluate the performance of the AI-Scientist using different problems:
### 1. SokanaAI Scientist: https://sakana.ai/ai-scientist/
import dspy
import asyncio
from httpx_aiohttp import HttpxAiohttpClient
from tools.file_reader import doc_reader
import pdb
# def fetch():
#       async with HttpxAiohttpClient() as client:
#             response = await client.get("https://encode.io")
#             print(response)

#       asyncio.run(main())

class Trajectory(dspy.Signature):
      paragraph = dspy.InputField(desc="Part of the reseach article")
      problem:str = dspy.OutputField(desc='the formulation of problem that is being solve in the paragraph or article')
      method:str = dspy.OutputField()
      result:str = dspy.OutputField()

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
            self.trajectories = dspy.ChainOfThought(signature = Trajectory)
            self.tool = doc_reader
            return
      
      def build_rag(self, query:str):
            url = "https://arxiv.org/pdf/2412.09871"
            doc = self.tool(url)
            return [doc]
      
      def forward(self, prompt:str):
            tr_list = []
            paper = self.build_rag(query=prompt)
            for section in paper.split("##"):
                  trl_list.append(self.trajectories(paragraph=section))
            return tr_list
      
if __name__ == "__main__":

      import os
      os.environ["MISTRAL_API_KEY"] = "sBEjabQFOosPqTWOrBkJKKXFQK9RHCha"
      lm = dspy.LM("mistral/ministral-8b-2512")
      dspy.configure(lm=lm)
      idea = "How to create the byte free input layer for neural network "
      rag_agent = RAG()
      #Search agent:
      Trajectory = rag_agent.forward(idea)
      print(Trajectory)
