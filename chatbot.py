import asyncio
from dataclasses import dataclass
import chainlit as cl
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    input_guardrail,
    set_tracing_disabled,
    set_default_openai_api,
    set_default_openai_client,
    function_tool,
    RunContextWrapper,
    handoff,
)
from my_secrets import Secrets
from typing import cast
import json
from openai.types.responses import ResponseTextDeltaEvent
import requests
from rich import print
from chainlit.input_widget import Select, Switch
from user_settings import UserSettings
from pydantic import BaseModel

mySecrets = Secrets()
external_client = AsyncOpenAI(
    base_url=mySecrets.gemini_api_url, api_key=mySecrets.gemini_api_key
)

set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


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


@dataclass
class Developer:
    name: str
    city: str
    country: str


@function_tool
@cl.step(type="Get Author Details Tool")
def get_author_details(wrapper: RunContextWrapper[Developer]):
    """Returns Developer details aka Author details."""
    return f"The developer is {wrapper.context.name}, based in {wrapper.context.city}, {wrapper.context.country}."


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


@cl.set_starters
async def starters():
    """user_key = "demo_user_123"
    user_settings = UserSettings(user_key)
    settings = user_settings.settings"""

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
            name="GPT-2.5 Flash",
            markdown_description="The underlying LLM model is **GPT-2.5 Flash**.",
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
            Switch(
                id="enable_weather",
                label="Enable Weather Tool",
                initial=settings.get("enable_weather", True),
            ),
            Switch(
                id="enable_student_info",
                label="Enable Student Info Tool",
                initial=settings.get("enable_student_info", True),
            ),
            Switch(
                id="enable_essay",
                label="Enable Essay Tool",
                initial=settings.get("enable_essay", True),
            ),
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
    developer = Developer(city="Lahore", country="Pakistan", name="Asad Waheed")
    try:
        print(f"[BEFORE RUN] Agent: {agent.name}")
        result = Runner.run_streamed(
            starting_agent=agent, input=chat_history, context=developer
        )
        print(
            f"[AFTER RUN] Last agent (from result): {getattr(result, 'last_agent', 'N/A')}"
        )
        response_message = cl.Message(content="")
        first_response = True
        got_response = False

        async with asyncio.timeout(20):  # Python 3.11+
            async for chunk in result.stream_events():
                # print(f"[EVENT CHUNK] {chunk.type}")
                if chunk.type == "raw_response_event" and isinstance(
                    chunk.data, ResponseTextDeltaEvent
                ):
                    if first_response:
                        await mythinking.remove()
                        await response_message.send()
                        first_response = False
                    await response_message.stream_token(chunk.data.delta)
                    got_response = True

        if not got_response:
            await mythinking.remove()
            await cl.Message(
                content="🤖 Agent did not respond. Please try again or say 'back to support' to return."
            ).send()
            return
        final_agent = result.last_agent
        if final_agent and final_agent != agent:
            print(f"✅ Switched to agent: {final_agent.name}")
            cl.user_session.set("agent", final_agent)
        chat_history.append({"role": "assistant", "content": response_message.content})
        cl.user_session.set("chat_history", chat_history)
        await response_message.update()
    except InputGuardrailTripwireTriggered as e:
        await mythinking.remove()
        await cl.Message(content="Please provide billing related queries only.").send()
    except asyncio.TimeoutError:
        await mythinking.remove()
        await cl.Message(content="⏱️ Timed out. Agent took too long to respond.").send()
    except Exception as e:
        response_message.content = f"An error occurred: {e}. Please try again later."
        await response_message.update()


@cl.on_chat_end
def end():
    chat_history = cl.user_session.get("chat_history") or []
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)


# Define the output model for the guardrail agents
class BillingOutput(BaseModel):
    is_billing_related: bool
    reasoning: str


class RefundOutput(BaseModel):
    is_refund_related: bool
    reasoning: str


# Define the input guardrail function
@input_guardrail
async def billing_input_relevance_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list,
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)
    final_output = result.final_output_as(BillingOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_billing_related,
    )


# Create the input guardrail agent to check if input is PIAIC-related
input_guardrail_agent = Agent(
    name="Billing_Relevance_Check",
    instructions=(
        """You are a guardrail agent responsible for identifying if a user's input is related to billing topics. This includes, but isn't limited to, inquiries about invoices, payments, charges, subscriptions, pricing, or account statements. Please note: Inquiries specifically about refunds are handled by a separate agent and should not be flagged as billing-related by this agent.

Your task is to thoroughly analyze the user's input and determine its relevance to any billing-related matter, excluding refunds.

Return a structured output containing two key pieces of information:

is_billing_related: A boolean value (true if related, false if not).

reasoning: A clear and concise explanation for your decision, referencing specific keywords or phrases from the user's input that led to your conclusion."""
    ),
    output_type=BillingOutput,
    model=OpenAIChatCompletionsModel(
        openai_client=external_client, model=mySecrets.gemini_api_model_2
    ),
)


def build_agent(settings: dict) -> Agent:
    tools = []
    chat_profile = cl.user_session.get("chat_profile")
    print(f"Chat profile: {chat_profile}")
    profileModel = mySecrets.gemini_api_model
    if chat_profile == "GPT 2.0 Flash":
        profileModel = mySecrets.gemini_api_model_2
    else:
        profileModel = mySecrets.gemini_api_model

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
    tools.append(get_author_details)
    instructions = """""
        You are a friendly and informative assistant. You can answer general questions and provide specific information.
        * For **weather inquiries**, you may fetch and share the current weather.
        * For **student-related queries**, you can retrieve details using the student ID.
        * For **essay writing**, you can retrieve an essay on a given topic.
        * For **Developer details**, you can provide developer details.
        * You are also a triage agent who can handoff to billing and refund agents.
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
        * For **Developer details**, you can provide developer details.
        * You are also a triage agent who can handoff to billing and refund agents.
        * Use tools **only when necessary**, not by default.
        * If a question falls outside essay writing, weather or student information, provide a helpful general response or ask for clarification.
        * If you're unsure of the answer, say "I don't know" or ask for more details.
        """
    """authorAgent: Agent = Agent[Developer](
        name="Assistant",
        tools=[get_author_details],
        instructions="If one ask for developer details use the respective tool.",
        model=profileModel,
    )"""

    def on_handoff(agent: Agent, ctx: RunContextWrapper[None]):
        agent_name = agent.name
        print(f"Handing off to {agent_name}...")
        # Send a more visible message in the chat
        cl.user_session.set("agent", agent)
        asyncio.create_task(
            cl.Message(
                content=f"🔄 **Handing off to {agent_name}...**\n\nI'm transferring your request to our {agent_name.lower()} who will be able to better assist you.",
                author="System",
            ).send()
        )

    billing_agent = Agent(
        name="Billing Agent",
        instructions="""You are the billing agent. Answer all billing-related questions helpfully.""",
        input_guardrails=[billing_input_relevance_guardrail],
        model=profileModel,
    )

    refund_agent = Agent(
        name="Refund Agent",
        instructions="You are the refund agent. Help users with refund-related queries.",
        model=profileModel,
    )

    return Agent[Developer](
        name="Assistant",
        instructions=instructions,
        model=profileModel,
        tools=tools,
        handoffs=[
            handoff(
                billing_agent, on_handoff=lambda ctx: on_handoff(billing_agent, ctx)
            ),
            handoff(refund_agent, on_handoff=lambda ctx: on_handoff(refund_agent, ctx)),
        ],
    )


async def initialize_agent(settings: dict):
    agent = build_agent(settings)
    cl.user_session.set("agent", agent)
