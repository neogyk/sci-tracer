# import dspy
# #from scitracer.tools.file_reader import doc_reader
# #from scitracer.tools.pub_search import arxiv_search, scolars_search
# import pdb
# #1. Using the question find the papers
# #2. Parse the papers and decompose it into the sequence of the problem -> solution -> results/conclustion
# #3. Having the list of the trajectories. Create own instructions
# #4. Follow the instructions to perform the task.

# class Trajectory(dspy.Signature):
#       question = dsqpy.InputField(desc="The problem or question that is necessary to solve")
#       paragraph = dspy.InputField(desc="Part of the reseach article")
#       problem:str = dspy.OutputField(desc='the formulation of problem that is being solve in the paragraph or article')
#       method:str = dspy.OutputField("The instuction that author or other scientist or people proceeded in order to solve this problem. In could be the methodology or algorithm of list of instruction")
#       result:str = dspy.OutputField("The result obtained by solving this problems")

# class RAG(dspy.Module):
#       """
#       The RAG Module provides the discovery trajectories for given idea, question or hypothesis.
#       Please chunk the text of the document and decompose it into the sequence of the
#       problem - method of the problem - solution and cosequences. This discovery trajectory for each of the paper
#       will be used for the research planning.
#       """
#       def __init__(self):
#             self.trajectories = dspy.Predict(Trajectory)
#             self.tool = [arxiv_search, scolars_search, doc_reader]
#             self.context = dspy.React("question->paper", tools=self.tools)
#             return

#       def forward(self, prompt:str):
#             tr_list = []
#             pdb.set_trace()
#             papers = self.build_rag(query=prompt)
#             pdb.set_trace()
#             #Form the trajectory
#             for paper in papers:
#                   for section in tqdm.tqdm(paper.split("##")):
#                         if len(section)==0:
#                               continue
#                         try:
#                               tr_list.append(self.trajectories(paragraph=section))
#                         except Exception:
#                               pass
#             return tr_list


# if __name__ == "__main__":
#       import os
#       os.environ["MISTRAL_API_KEY"] = "sBEjabQFOosPqTWOrBkJKKXFQK9RHCha"
#       lm = dspy.LM("mistral/ministral-8b-2512")
#       dspy.configure(lm=lm)
#       idea = "How to create the byte free input layer for neural network "
#       rag_agent = RAG()
#       #Search agent:
#       tr_list = rag_agent.forward(idea)
#       print(tr_list)
#       pdb.set_trace()
