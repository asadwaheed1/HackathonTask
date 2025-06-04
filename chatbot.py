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
from chainlit.input_widget import Select, Switch
from user_settings import UserSettings
mySecrets = Secrets()


@function_tool("weather_tool")
@cl.step(type="Get Weather Tool")
async def get_weather(location: str) -> str:
    """
    Retrieves current weather information for a specified location.

    This function makes an asynchronous API call to fetch real-time weather data
    including temperature, weather conditions, wind speed, humidity, and UV index
    for the given location.

    Args:
        location (str): The location for which to fetch weather data. Can be a city name,
                       coordinates, or other location identifier supported by the weather API.

    Returns:
        str: A formatted string containing comprehensive weather information including:
             - Location details (name, region, country)
             - Current date and time
             - Temperature in Celsius and "feels like" temperature
             - Weather condition description
             - Wind speed (km/h) and direction
             - Humidity percentage
             - UV index

             If the API request fails, returns an error message indicating the failure.

    Raises:
        This function handles HTTP errors internally and returns error messages as strings
        rather than raising exceptions.

    Example:
        >>> weather = await get_current_weather("London")
        >>> print(weather)
        Current weather in London, England, United Kingdom as of 2023-10-15 14:30 is 18°C (Partly cloudy), feels like 17°C, wind 15 km/h SW, humidity 65% and UV index is 4.
    """
    result = requests.get(
        f"{mySecrets.weather_api_url}/current.json?key={mySecrets.weather_api_key}&q={location}"
    )
    if result.status_code != 200:
        return f"Failed to fetch weather data for {location}. Please try again later."
    data = result.json()
    return f"Current weather in {data['location']['name']}, {data['location']['region']}, {data['location']['country']} as of {data['current']['last_updated']} is {data['current']['temp_c']}°C ({data['current']['condition']['text']}), feels like {data['current']['feelslike_c']}°C, wind {data['current']['wind_kph']} km/h {data['current']['wind_dir']}, humidity {data['current']['humidity']}% and UV index is {data['current']['uv']}."


@function_tool("student_info_tool")
@cl.step(type="Get Student Info Tool")
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


import chainlit as cl
from chainlit.context import get_context, ChainlitContextException

@cl.set_starters
async def starters():
    """  user_key = "demo_user_123"
    user_settings = UserSettings(user_key)
    settings = user_settings.settings """

    starters_list = []

    starters_list.append(
            cl.Starter(
                label="Get Current Weather",
                message="Fetch the current weather for a specified location.",
                icon="/public/weather.svg",
            )
        )

    starters_list.append(
            cl.Starter(
                label="Get Student Info",
                message="Retrieve information about a student using their ID.",
                icon="/public/student.svg",
            )
        )

    starters_list.append(
            cl.Starter(
                label="Write an Essay",
                message="Generate a 1000-word essay on a given topic.",
                icon="/public/article.svg",
            )
        )

    starters_list.append(
        cl.Starter(
            label="Explore General Questions",
            message="Find answers to the given questions.",
            icon="/public/question.svg",
        )
    )

    return starters_list




@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="GPT 2.0 Flash",
            markdown_description="The underlying LLM model is **GPT-2.0 Flash**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GPT-1.5",
            markdown_description="The underlying LLM model is **GPT-1.5**.",
            icon="https://picsum.photos/250",
        ),
    ]


@cl.on_settings_update
async def on_settings_update(settings: dict):
    cl.user_session.set("chat_settings", settings) 
    settings = await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Chat Mode",
                values=["Casual", "Technical"],
                initial_value=settings.get("mode", "Technical"),
            ),
            Switch(id="enable_weather", label="Enable Weather Tool", initial=settings.get("enable_weather", True)),
            Switch(
                id="enable_student_info", label="Enable Student Info Tool", initial=settings.get("enable_student_info", True)
            ),
            Switch(id="enable_essay", label="Enable Essay Tool", initial=settings.get("enable_essay", True)),
        ]
    ).send()
    user_key = "demo_user_123"
    user_settings = UserSettings(user_key)
    user_settings.settings = settings
    user_settings.save_settings()
    await initialize_agent(settings)


@cl.on_chat_start
async def start():
    user_key = "demo_user_123"
    print(f"Chat started for user: {user_key}")
    user_settings = UserSettings(user_key)
    saved_settings = user_settings.settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Chat Mode",
                values=["Casual", "Technical"],
                initial_value=saved_settings.get("mode", "Technical"),
            ),
            Switch(
                id="enable_weather",
                label="Enable Weather Tool",
                initial=saved_settings.get("enable_weather", True),
            ),
            Switch(
                id="enable_student_info",
                label="Enable Student Info Tool",
                initial=saved_settings.get("enable_student_info", True),
            ),
            Switch(
                id="enable_essay",
                label="Enable Essay Tool",
                initial=saved_settings.get("enable_essay", True),
            ),
        ]
    ).send()
    
    
    set_tracing_disabled(True)

   
    cl.user_session.set("chat_settings", settings)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("user_settings", user_settings)  
    await initialize_agent(settings)


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
        response_message.content = f"An error occurred: {e}. Please try again later."
        await response_message.update()


@cl.on_chat_end
def end():
    chat_history = cl.user_session.get("chat_history") or []
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)


def build_agent(settings: dict) -> Agent:
    tools = []
    chat_profile = cl.user_session.get("chat_profile")
    print(f"Chat profile: {chat_profile}")
    profileModel = mySecrets.gemini_api_model
    if chat_profile == "GPT 2.0 Flash":
        profileModel = mySecrets.gemini_api_model_2
    else:
        profileModel = mySecrets.gemini_api_model
    external_client = AsyncOpenAI(
        base_url=mySecrets.gemini_api_url, api_key=mySecrets.gemini_api_key
    )
    essay_agent = Agent(
        name="Essay Writer",
        instructions="You are an expert essay writer. Write a detailed 1000 word essay on the given topic.",
        model=OpenAIChatCompletionsModel(
            openai_client=external_client, model=profileModel
        ),
    )

    if settings.get("enable_weather"):
        tools.append(get_weather)

    if settings.get("enable_student_info"):
        tools.append(get_student_info)

    if settings.get("enable_essay"):
        tools.append(
            essay_agent.as_tool(
                tool_name="essay_writer_tool",
                tool_description="Write a detailed 1000-word essay on the given topic.",
            )
        )
    instructions = """""
        You are a friendly and informative assistant. You can answer general questions and provide specific information.
        * For **weather inquiries**, you may fetch and share the current weather.
        * For **student-related queries**, you can retrieve details using the student ID.
        * For **essay writing**, you can retrieve an essay on a given topic.
        * Use tools **only when necessary**, not by default.
        * If a question falls outside essay writing, weather or student information, provide a helpful general response or ask for clarification.
        * If you're unsure of the answer, say "I don't know" or ask for more details.
        """
    if settings.get("mode") == "Casual":
        instructions = """""
        You are a friendly and informative assistant. You answer all questions in casual tone. You can answer general questions and provide specific information.
        * For **weather inquiries**, you may fetch and share the current weather.
        * For **student-related queries**, you can retrieve details using the student ID.
        * For **essay writing**, you can retrieve an essay on a given topic.
        * Use tools **only when necessary**, not by default.
        * If a question falls outside essay writing, weather or student information, provide a helpful general response or ask for clarification.
        * If you're unsure of the answer, say "I don't know" or ask for more details.
        """

    return Agent(
        name="Assistant",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            openai_client=external_client, model=profileModel
        ),
        tools=tools,
    )
    
async def initialize_agent(settings: dict):
    agent = build_agent(settings)
    cl.user_session.set("agent", agent)