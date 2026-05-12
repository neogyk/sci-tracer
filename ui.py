from rich.prompt import Prompt
#cretat

def setup():
    #TODO inspect the stored session from s3 storage and propose user to use it
    name = Prompt.ask("Name of session", 
                      choices=["Default session"], 
                      default="New session")
    
    #Get the list of the existing environment
    env = Prompt.ask("Define the environment", 
                     choices=["conda", "pixi"])

    gitclient = Prompt.ask("Define the environment",
                            choices=["conda", "pixi"])
    
    taskflow = Prompt.ask("Adress of the taskflow server", 
                          choices=[""])
    
    verification_protocol = Prompt.ask("Define the environment",
                                        choices=["z3", "pixi"])
    
    config_protocol = Prompt.ask("Remote server configuration",
                                        choices=[".comf.yaml"])
    
    project_name = Prompt.ask("What is the project name")
    
    # if llm_name is not "ollama":

    # else:

    # return

if __name__ == "__main__":
    setup()
