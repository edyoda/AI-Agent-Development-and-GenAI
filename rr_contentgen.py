import asyncio
import os
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TaskResult
from fastapi import FastAPI, HTTPException, BackgroundTasks

import httpx
from pydantic import BaseModel

app = FastAPI()

# Add this Pydantic model to define the request body
class InterviewRequest(BaseModel):
    topic: str
    subtopics: list[str]
    session_details_meta_id: str



async def generate_qa(topic, subtopics):
    # Create an OpenAI model client.

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        temperature=1,
    )

    # Define agents with updated AutoGen 0.8.5 syntax
    topic_understanding_agent = AssistantAgent(
        name="Topic_Understanding_Agent",
        system_message="""
        You are an expert in analyzing interview topic scopes for senior professionals (10+ years experience).
        Your task is to:
        1. Refine and validate the given topic and its sub-topics.
        2. Expand sub-topics that are vague or too broad into more specific and distinct concepts.
        3. For each sub-topic, classify it into one or more of the following categories:
        - Technical Depth
        - Business Impact
        - Leadership Dimension

        Your output should be a structured JSON like this:
        {
        "topic": "<refined topic>",
        "sub_topics": [
                {
                "name": "<sub-topic-1>",
                "expanded": ["<specific area 1>", "<specific area 2>", ...],
                "categories": ["Technical Depth", "Business Impact"]
                },
                ...
        ]
        }
        """,
        model_client=model_client,
    )

    knowledge_mapping_agent = AssistantAgent(
        name="Knowledge_Mapping_Agent",
        system_message="""
            You are a senior-level capability architect responsible for defining expert-level knowledge expectations.

            Given a sub-topic with its expanded areas and category tags, describe the kind of deep knowledge and practical expertise a professional with 10+ years of experience should have.

            For each sub-topic, return:
            - A summary of the depth of understanding expected
            - Real-world responsibilities typically handled
            - Architectural or leadership decisions made in that area
            - Industry challenges or patterns commonly dealt with

            Output format:
            {
            {
            "sub_topic": "<name1>",
            "knowledge_scope": [
            {
            "area": "<expanded area>",
            "expertise": "<depth of knowledge expected>",
            "responsibilities": ["<responsibility 1>", "<responsibility 2>"],
            "typical_decisions": ["<decision 1>", "<decision 2>"],
            "common_challenges": ["<challenge 1>", "<challenge 2>"]
            },
            ...
            ]
            },
            {
            "sub_topic": "<name2>",
            "knowledge_scope": [
            {
            "area": "<expanded area>",
            "expertise": "<depth of knowledge expected>",
            "responsibilities": ["<responsibility 1>", "<responsibility 2>"],
            "typical_decisions": ["<decision 1>", "<decision 2>"],
            "common_challenges": ["<challenge 1>", "<challenge 2>"]
            },
            ...
            ]
            },
            ...
            }
            """,
        model_client=model_client,
        )

    qa_generator_agent = AssistantAgent(
        name="QA_Generator_Agent",
        system_message="""
            You are an expert interview panelist specializing in evaluating candidates with 10+ years of experience.

            For each sub-topic area provided with knowledge mapping, generate a list of interview questions and expert-level answers.

            Guidelines:
            - Focus on senior-level depth: trade-offs, design patterns, leadership calls, stakeholder impact.
            - Questions should be open-ended, scenario-based, or architecture-focused.
            - Avoid trivia or low-level technical questions.
            - Answers should reflect maturity, breadth, and clarity expected from a seasoned professional. Add examples where relevant.

            Output format:
            {
            [
                {
                "question": "<interview question>",
                "answer": "<expert-level answer>"
                },
                {
                "question": "<interview question>",
                "answer": "<expert-level answer>"
                },
                ...
            ]
            }
            """,
        model_client=model_client,
        )

    json_formatter_agent = AssistantAgent(
        name="JSON_Formatter_Agent",
        system_message="""
             "You the a perfect JSON formatter. Your outcome is perfectly parseable JSON by json.loads().

             Remove or replace any charectes which might impact json parsing the message. Give the outcome strictly in the below mentioned format

             Output format:
            {
            "questions_and_answers":
            [
                {
                "question": "<interview question>",
                "answer": "<expert-level answer>"
                },
                {
                "question": "<interview question>",
                "answer": "<expert-level answer>"
                },
                ...
            ]
            }
            """,
        model_client=model_client,
        )



    # Create the primary agent.
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedbacks are addressed.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")
    max_msg_termination = MaxMessageTermination(max_messages=5)

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat([topic_understanding_agent, knowledge_mapping_agent, qa_generator_agent, json_formatter_agent], termination_condition=max_msg_termination)

    #result = await team.run(task="Write a short poem about the fall season.")
    #print(result)

       # Process messages as they're generated
    prompt = f"""
        TOPIC: {topic}
        SUBTOPICS: {', '.join(subtopics)}
        TARGET: Senior professionals (15-20 years experience)

        Output format:
        - 10 Technical Depth Questions
        - 6 Strategic Leadership Questions
        - 2 Behavioral Scenario
        - 2 Innovation/Transformation Question
        """

    async for message in team.run_stream(task=prompt):
        if isinstance(message, TaskResult):
            print("\nStop Reason:", message.stop_reason)
        else:
            print(f"\n{message.source}: {message.content}")
    
    return message

def get_last_qa_generator_message(task_result):
    # Iterate through messages in reverse order to find the last one from QA_Generator_Agent
    for message in reversed(task_result.messages):
        if isinstance(message, TextMessage) and message.source == 'JSON_Formatter_Agent':
            return message.content
    return None  # Return None if no matching message is found

'''
# Run the async main function
if __name__ == "__main__":
    result = asyncio.run(generate_qa("AWS","[Storage,Compute,EC2]"))

    print("***********************")
    print(get_last_qa_generator_message(result))
'''

async def async_contentgen_task(topic, subtopics, meta_id):

    print(topic, subtopics, meta_id)

    result = await generate_qa(topic,subtopics)

    content = get_last_qa_generator_message(result)
    print(f"Background task finished!")

    try:
        async with httpx.AsyncClient() as client:
                response = await client.post(
                        "http://13.200.116.100:8000/api/v1/studycontentgen/callfromlambdadetails/",
                        json={
                            "status": "success",
                            "topic": topic,
                            "subtopics": subtopics,
                            "meta_id": meta_id,
                            "result": content
                            },
                    timeout=30.0
                )
                print(f"Webhook response status: {response.status_code}")
    except Exception as e:
        print(f"Failed to send webhook: {str(e)}")



@app.post("/generate_interview_questions")
async def generate_interview_questions(background_tasks: BackgroundTasks, request: InterviewRequest):
    try:
        background_tasks.add_task( async_contentgen_task,
                topic = request.topic,
                subtopics = request.subtopics,
                meta_id = request.session_details_meta_id)

        return {
            "status": "success",
            "result": "Async processing started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

