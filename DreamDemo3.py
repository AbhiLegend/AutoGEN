import os
import time
import pandas as pd
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import MessageTextContent
from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor

# -----------------------------------------------------------------------------
# Setup Azure AI Project Client
# -----------------------------------------------------------------------------

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str="eastus.api.azureml.ms;3f01ab49-a56f-4ee7-97fa-d23155156b42;rg-mcfretail-demo-dev;prj-mcfr-agents-exp",
)

# -----------------------------------------------------------------------------
# Enable Azure Monitor Tracing (if enabled for your project)
# -----------------------------------------------------------------------------
application_insights_connection_string = project_client.telemetry.get_connection_string()
if not application_insights_connection_string:
    print("Application Insights was not enabled for this project.")
    print("Enable it via the 'Tracing' tab in your AI Foundry project page.")
    exit()

configure_azure_monitor(connection_string=application_insights_connection_string)

scenario = os.path.basename(__file__)
tracer = trace.get_tracer(__name__)

# -----------------------------------------------------------------------------
# AgentHandler Class for Reusable Agents
# -----------------------------------------------------------------------------
class AgentHandler:
    def __init__(self, client):
        self.client = client

    def run_task(self, agent_name, agent_instructions, user_message):
        # Create an agent with the specified role/instructions
        agent = self.client.agents.create_agent(
            model="gpt-4o",
            name=agent_name,
            instructions=agent_instructions
        )
        print(f"Created agent '{agent_name}', agent ID: {agent.id}")

        # Create a thread for this task
        thread = self.client.agents.create_thread()
        print(f"Created thread, thread ID: {thread.id}")

        # Send user message
        message = self.client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=user_message
        )
        print(f"Created message, message ID: {message.id}")

        # Create a run and poll until completion
        run = self.client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = self.client.agents.get_run(thread_id=thread.id, run_id=run.id)
            print(f"Run status: {run.status}")

        # Retrieve all messages; the assistant's reply will be the last message
        messages = self.client.agents.list_messages(thread_id=thread.id)
        assistant_reply = messages['data'][0]['content'][0]['text']['value']
        print(f"Agent '{agent_name}' reply: {assistant_reply}")

        # Clean up the agent if you don't need it further
        self.client.agents.delete_agent(agent.id)
        print(f"Deleted agent '{agent_name}'")
        return assistant_reply

# -----------------------------------------------------------------------------
# Example: Calling an Agent Based on User Input
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    with tracer.start_as_current_span(scenario):
        with project_client:
            agent_handler = AgentHandler(project_client)

            while True:
                # Get user input to start an agent
                user_input = input("Enter a phrase to start an agent (or type 'exit' to quit): ").strip()

                if user_input.lower() == 'exit':
                    print("Exiting the program.")
                    break

                # Define agent-specific instructions based on user input
                if "validate" in user_input.lower():
                    agent_name = "Data_Validation_Agent"
                    agent_instructions = """
                    You validate the provided data for any inconsistencies or missing values.
                    """
                    user_message = "Here is the data: {'Revenue': 100, 'Cost': 50, 'Profit': None}. Please validate it."

                elif "summarize" in user_input.lower():
                    agent_name = "Summary_Generation_Agent"
                    agent_instructions = """
                    You summarize the financial report for executive presentations.
                    """
                    user_message = "Summarize the financial report for executives."

                elif "review" in user_input.lower():
                    agent_name = "Accuracy_Review_Agent"
                    agent_instructions = """
                    You check the financial reports for accuracy and consistency.
                    """
                    user_message = "Please review the accuracy of the financial report."

                else:
                    print("No agent matched the input phrase. Try 'validate', 'summarize', or 'review'.")
                    continue

                # Call the agent
                response = agent_handler.run_task(
                    agent_name=agent_name,
                    agent_instructions=agent_instructions,
                    user_message=user_message
                )

                # Print the response
                print("Agent response:", response)
