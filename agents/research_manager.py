import dspy


class ResearchManage(dspy.Module):
    """
    Research manager creates than plan and delegate the actions to the subagent that is created by following the plan
    of discovery.
    """

    def __init__(self):
        self.trajectories = dspy.Predict(Trajectory)
        self.tool = doc_reader
        return

    # def build_rag(self, query:str):
    #       url = "https://arxiv.org/pdf/2412.09871"
    #       url = "https://uni-tuebingen.de/fileadmin/Uni_Tuebingen/Fakultaeten/MathePhysik/Institute/IAP/Forschung/MOettel/Geburt_QM/schrodinger_AnnPhys_386_109_1926.pdf"
    #       url = "https://inters.org/files/einstein1905_photoeff.pdf"
    #       url = "https://www.jetp.ras.ru/cgi-bin/dn/e_020_06_1490.pdf"
    #       doc = self.tool(url)
    #       return doc

    def forward(self, prompt: str):
        tr_list = []
        paper = self.build_rag(query=prompt)
        for section in tqdm.tqdm(paper.split("##")):
            if len(section) == 0:
                continue
            try:
                tr_list.append(self.trajectories(paragraph=section))
            except Exception:
                pass
        return tr_list
