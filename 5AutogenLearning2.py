from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import os

import os

# Configure a teachable agent
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY") # Replace with your actual API key
    }
]

# Create agents with better configuration
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "seed": 42,
        "temperature": 0.7,
    },
    system_message="""You are a helpful AI assistant capable of learning. 
    You remember user preferences, can learn new procedures, incorporate feedback, 
    and apply knowledge across domains. Respond TERMINATE when the task is complete."""
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,  # Disable code execution for these demos
)

# 1. Memory Integration Example - Fixed version
def demonstrate_memory_integration():
    print("\n=== 1. Memory Integration Demonstration ===")
    
    # First interaction - agent stores information
    user_proxy.initiate_chat(
        assistant,
        message="My name is John and I prefer responses in French. Please acknowledge this with 'TERMINATE'.",
    )
    
    # Second interaction - agent recalls previous information
    user_proxy.initiate_chat(
        assistant,
        message="What is my name and in what language should you respond? Respond with 'TERMINATE' when done.",
    )

# 2. Skill Acquisition Example - Fixed version
def demonstrate_skill_acquisition():
    print("\n=== 2. Skill Acquisition Demonstration ===")
    
    # Teach the agent a new procedure
    user_proxy.initiate_chat(
        assistant,
        message="""Learn this procedure for calculating Fibonacci numbers:
        1. Start with 0 and 1
        2. Each subsequent number is the sum of the two preceding ones
        3. Continue until you reach the desired sequence length
        
        Now show me the first 10 Fibonacci numbers using this procedure. 
        Respond with 'TERMINATE' when done.""",
    )
    
    # Later application of the learned skill
    user_proxy.initiate_chat(
        assistant,
        message="Using the Fibonacci procedure you learned, calculate up to the 15th number. Respond with 'TERMINATE' when done.",
    )

# 3. Feedback Incorporation Example - Fixed version
def demonstrate_feedback_incorporation():
    print("\n=== 3. Feedback Incorporation Demonstration ===")
    
    # Initial attempt with feedback
    user_proxy.initiate_chat(
        assistant,
        message="Explain quantum computing to a 5-year-old. Respond with 'TERMINATE' when done.",
    )
    
    # Provide feedback and observe adaptation
    user_proxy.initiate_chat(
        assistant,
        message="That was too complex. Simplify it further using only animal analogies. Respond with 'TERMINATE' when done.",
    )
    
    # Verify the agent adapted its approach
    user_proxy.initiate_chat(
        assistant,
        message="Now explain blockchain to a 5-year-old using animal analogies. Respond with 'TERMINATE' when done.",
    )

# 4. Transfer Learning Example - Fixed version
def demonstrate_transfer_learning():
    print("\n=== 4. Transfer Learning Demonstration ===")
    
    # Teach a concept in one domain
    user_proxy.initiate_chat(
        assistant,
        message="""Learn this pattern: 
        When analyzing time-series data:
        1. Check for seasonality
        2. Remove trends
        3. Normalize the data
        4. Apply appropriate forecasting models
        Acknowledge with 'TERMINATE'.""",
    )
    
    # Apply similar pattern in a different domain
    user_proxy.initiate_chat(
        assistant,
        message="""How would you approach analyzing spatial geographical data 
        using a similar methodology to what you learned about time-series?
        Respond with 'TERMINATE' when done.""",
    )

# Run demonstrations sequentially with clear termination
demonstrate_memory_integration()
demonstrate_skill_acquisition()
demonstrate_feedback_incorporation()
demonstrate_transfer_learning()

print("\nAll demonstrations completed successfully!")
