### Build an OpenAI Agent SDK-Based Chatbot with Chainlit Hackathon Task

**Assalamualikum! To run the project you must have python installed and uv installed.**


## How to run

- Make sure you have installed all dependencies mentioned in pyproject.toml file
- Create a .env file in project root and add required strings for reference see my_secrets fil (Important)
- Run the project with "chainlit run chatbot.py"
- chat_history.json file is generated in project root directory

## Screenshots
![App Screenshot](https://github.com/user-attachments/assets/5124a5dc-b398-4832-a5fc-a92f23431e46)

## 📌 Chatbot Overview

This chatbot is a smart, browser-based assistant built using **Chainlit** and the **OpenAI Agent SDK**. It offers a rich interactive experience with the following capabilities:

### ✅ Core Features
- **Real-time Streaming Responses**: Responses are streamed live, character-by-character.
- **Complete Chat History**: Maintains the full conversation history.
- **Auto-Save**: Saves the chat to a `chat_history.json` file at the end of each session.
- **Starter Prompts**: Displays pre-defined suggestions to guide users at the start of each session.
- **"Thinking..." Placeholder**: Shown while the assistant generates a response.
- **Visible Tool Usage**: Any tool invocation appears directly in the chat interface.

### 🛠️ Integrated Tools
- **🌤️ Weather Tool**: Connects to an external API to provide real-time weather data.
- **🎓 Student Info Tool**: Retrieves student data from a built-in dataset.
- **📝 Essay Writer**: A custom agent-based tool that generates 1000-word essays on user-provided topics.

### ⚙️ Customization with Chat Profiles
Users can customize their chat experience using **Chainlit Chat Profiles**, which allow:
- **Model Selection**: Choose between LLMs like `GPT-2.0 Flash` or `GPT-1.5`.
- **Chat Mode**: Select between `Casual` and `Technical` interaction styles.
- **Tool Toggle**: Enable or disable specific tools based on user preference.

> 🔒 **Persistent Settings**: User settings are saved and persist across future chat sessions.

---

This setup creates a flexible, intelligent chatbot that adapts to user preferences while providing real-time, helpful, and tool-augmented responses.




