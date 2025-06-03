import chainlit as cl
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
)
from my_secrets import Secrets
from typing import cast
import json
from openai.types.responses import ResponseTextDeltaEvent
import requests
from rich import print

mySecrets = Secrets()

@function_tool('student_info_tool')
async def get_student_info(student_id: int):
    """Fetch student information by ID."""
    students = {
        1: {"name": "Alice", "age": 20, "major": "Computer Science"},
        2: {"name": "Bob", "age": 22, "major": "Mathematics"},
        3: {"name": "Charlie", "age": 21, "major": "Physics"},
        4: {"name": "Diana", "age": 23, "major": "Chemistry"},
        5: {"name": "Ethan", "age": 19, "major": "Biology"},
    }
    student_info = students.get(student_id, None)
    if student_info:
        return f"Student ID: {student_id}, Name: {student_info['name']}, Age: {student_info['age']}, Major: {student_info['major']}"
    else:
        return f"No student found with ID {student_id}."



@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label = "Get Current Weather",
            message = "Fetch the current weather for a specified location.",
            icon= "/public/weather.svg",
        ),
        cl.Starter(
            label="Get Student Info",
            message="Retrieve information about a student using their ID.",
            icon="/public/student.svg",
        ),
        cl.Starter(
            label="Explore General Questions",
            message="Find answers to the given questions.",
            icon="/public/question.svg",
        ),
        cl.Starter(
            label="Write an Essay",
            message="Generate an 1000 words essay on a given topic.",
            icon="/public/article.svg",
        ),
    ]


@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        base_url=mySecrets.gemini_api_url, api_key=mySecrets.gemini_api_key
    )
    set_tracing_disabled(True)
    agent = Agent(
        name="Assistant",
        instructions="You are a friendly and informative assistant.",
        model=OpenAIChatCompletionsModel(
            openai_client=external_client, model=mySecrets.gemini_api_model
        ),
    )
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def main(msg: cl.Message):
    mythinking = cl.Message(content="Thinking...")
    await mythinking.send()

    agent = cast(Agent, cl.user_session.get("agent"))
    chat_history = cl.user_session.get("chat_history") or []
    chat_history.append({"role": "user", "content": msg.content})
    try:
        result = Runner.run_streamed(
            starting_agent=agent,
            input=chat_history,
        )
        response_message = cl.Message(content="")
        first_response = True

        async for chunk in result.stream_events():
            if chunk.type == "raw_response_event" and isinstance(
                chunk.data, ResponseTextDeltaEvent
            ):
                if first_response:
                    await mythinking.remove()
                    await response_message.send()
                    first_response = False
                await response_message.stream_token(chunk.data.delta)

        chat_history.append({"role": "assistant", "content": response_message.content})
        cl.user_session.set("chat_history", chat_history)
        await response_message.update()
    except Exception as e:
        response_message.content = (
            f"An error occurred: {e}. Please try again later."
        )
        await response_message.update()


@cl.on_chat_end
def end():
    chat_history = cl.user_session.get("chat_history") or []
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)
