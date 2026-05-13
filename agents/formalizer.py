import dspy

class HoareFormalizer(dspy.Signature):
    """Convert natural language requirements into a formal Hoare Triple {P} C {Q}."""
    requirement = dspy.InputField(desc="The human-readable requirement")
    precondition = dspy.OutputField(desc="Logical state before (P)")
    command = dspy.OutputField(desc="The action or code (C)")
    postcondition = dspy.OutputField(desc="Logical state after (Q)")

# Define the module
class FormalizeRequirement(dspy.Module):
    def __init__(self):
        super().__init__()
        self.engine = dspy.ChainOfThought(HoareFormalizer)

    def forward(self, text):
        return self.engine(requirement=text)

# Example Execution
formalizer = FormalizeRequirement()
res = formalizer("If the user is logged in and clicks delete, the file should be gone.")

print(f"Pre: {res.precondition}")
print(f"Cmd: {res.command}")
print(f"Post: {res.postcondition}")

#TODO tranform the res into the logic expression compatible with Lean or Z3

