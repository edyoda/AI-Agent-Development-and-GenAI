# Example demonstrating how AutoGen implements LLM theory with human input
import autogen
import os

# The system message implements role-based prompting theory
system_message = """You are a mathematics expert with specialization in calculus.
Follow these principles in your responses:
1. Break down complex problems into steps
2. Show your reasoning explicitly
3. Verify your answers with alternative methods when possible
4. Explain concepts using intuitive analogies
5. Highlight common misconceptions to avoid"""

# Creating an agent with the specialized system message
math_expert = autogen.AssistantAgent(
    name="MathExpert",
    system_message=system_message,
    llm_config={
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.2,  # Lower temperature for more precise reasoning
    }
)

# User proxy representing the human in the interaction
user_proxy = autogen.UserProxyAgent(
    name="Student",
    human_input_mode="ALWAYS",  # Changed to ALWAYS to get human input for each message
    code_execution_config={"work_dir": "math_workspace"}
)

# Get initial input from the user
print("\nWelcome to the Math Expert Assistant!")
print("Type 'exit' to end the conversation at any time.\n")
initial_message = input("What math problem would you like help with today?\n> ")

# Start the chat only if the user didn't immediately exit
if initial_message.lower() != 'exit':
    user_proxy.initiate_chat(
        math_expert,
        message=initial_message
    )
else:
    print("Goodbye!")
