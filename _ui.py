import dspy
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.layout import Layout

# --- 1. Setup DSPy Agent ---
# Replace this with your actual DSPy module/signature
class SimpleQA(dspy.Signature):
    """Answer questions concisely based on the provided context."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="a detailed and helpful response")

# Initialize the DSPy predictor
# Ensure you have dspy.settings.configure(lm=...) set up elsewhere
#agent = dspy.Predictor(SimpleQA)

agent = None
# --- 2. Setup Rich UI ---
console = Console()

def display_welcome():
    welcome_msg = """
    # 🤖 DSPy Agent CLI
    Type your message below to chat. 
    - Type `exit` or `quit` to stop.
    - Markdown formatting is supported!
    """
    console.print(Panel(Markdown(welcome_msg), border_style="blue"))

def run_cli():
    display_welcome()
    
    while True:
        # Get User Input
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except EOFError:
            break

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        # Show Loading Spinner while DSPy processes
        with Live(Spinner("dots", text="[bold yellow]Agent is thinking...[/bold yellow]"), transient=True):
            try:
                # Call the DSPy Agent
                response = agent(question=user_input)
                answer_text = response.answer
            except Exception as e:
                answer_text = f"**Error:** {str(e)}"

        # Print the Formatted Response
        console.print(
            Panel(
                Markdown(answer_text),
                title="[bold green]DSPy Agent[/bold green]",
                title_align="left",
                border_style="green",
                padding=(1, 2)
            )
        )

if __name__ == "__main__":
    run_cli()
