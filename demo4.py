import os
import time
from autogen import ConversableAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the base llm_config function
def create_llm_config(model, temperature):
    return {
        "model": model,
        "temperature": temperature,
        "api_key": os.environ.get("OPEN_AI_KEY"),
    }

# -----------------------------------------------------------------------------
# AgentHandler Class for Reusable Agents
# -----------------------------------------------------------------------------
class AgentHandler:
    def __init__(self, agent_name, system_message, llm_config):
        self.agent = ConversableAgent(
            name=agent_name,
            system_message=system_message,
            llm_config={"config_list": [llm_config]},
            human_input_mode="NEVER",  # Never ask for human input.
        )

    def run_task(self, target_agent, user_message, max_turns=2):
        result = self.agent.initiate_chat(
            target_agent,
            message=user_message,
            max_turns=max_turns
        )
        return result

# -----------------------------------------------------------------------------
# Example: Setting Up and Running Agents
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize agents
    validation_agent = AgentHandler(
        agent_name="Data_Validation_Agent",
        system_message="You validate the provided data for any inconsistencies or missing values.",
        llm_config=create_llm_config("gpt-4", 0.8),
    )

    summary_agent = AgentHandler(
        agent_name="Summary_Generation_Agent",
        system_message="You summarize the financial report for executive presentations.",
        llm_config=create_llm_config("gpt-4", 0.7),
    )

    review_agent = AgentHandler(
        agent_name="Accuracy_Review_Agent",
        system_message="You check the financial reports for accuracy and consistency.",
        llm_config=create_llm_config("gpt-4", 0.7),
    )

    # Agent logic based on user input
    while True:
        user_input = input("Enter a phrase to start an agent (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("Exiting the program.")
            break

        if "validate" in user_input.lower():
            target_agent = validation_agent.agent
            user_message = "Here is the data: {'Revenue': 100, 'Cost': 50, 'Profit': None}. Please validate it."

        elif "summarize" in user_input.lower():
            target_agent = summary_agent.agent
            user_message = "Summarize the financial report for executives."

        elif "review" in user_input.lower():
            target_agent = review_agent.agent
            user_message = "Please review the accuracy of the financial report."

        else:
            print("No agent matched the input phrase. Try 'validate', 'summarize', or 'review'.")
            continue

        # Execute the task and retrieve the result
        response = validation_agent.run_task(
            target_agent=target_agent,
            user_message=user_message,
        )

        # Display the response
        print("Agent response:", response)
