# The Council Instrumentality Project
The Council Instrumentality Project is a conversational agent designed to interact in Discord channels, retrieve relevant historical context, and generate responses based on past interactions. This project leverages Pinecone for vector storage and retrieval, and HuggingFace for encoding the conversation context and current query for retrieval.

It is currently setup and prompted to be a librarian of sorts for me and my friend's old group chat, however it could be easily modified to do a variety of tasks that any effective RAG chatbot could do.

This setup allows it to be free to use, end to end.

## Features
- Integration with Pinecone for vector-based retrieval.
- Uses HuggingFace Encoder for text embeddings.
- Generates context-aware responses based on previous Discord messages.
- Supports both public channels and direct messages.
- Automatically uses CUDA for embedding/response generation on Nvidia GPUs.
- Option for local response generation.

## Usage
To install dependencies, make sure you are in the repository's directory after cloning, then run:
   ```bash
   pip install -r requirements.txt
   ```

This requires API keys from Groq, Discord, and Pinecone. Use the [.env.example](./.env.example) file as a template, then rename it to .env after
putting in your personal API keys.

Right now, the bot strictly requires that you have a Pinecone Index with the information you want to retrieve embedded within.

Initially, we exported all of our messages from the old group chat in JSON format and ran it through [message_parser](./src/message_parser.py), which gave us our dataset in a text file. From there, we used [index_message](./src/index_message.py) to automatically vectorize, upsert, and generate metadata in the Pinecone Index.

You must create a discord bot using the developer portal, which is also where you will get your API key from. 
[Discord Developer Portal](https://discord.com/developers/applications)

Once the bot is running, invite it to a Discord server using your bot token. The bot will automatically listen to messages and generate responses based on the conversation context in both channels when mentioned and direct messages.

### Commands
- To mention the bot, simply type `@botname` in a channel, and it will retrieve the most relevant past conversations to assist with its response. If you are in a Direct Message with the bot, you do not have to mention it.

## Local Generation
[responsesLocal](./src/responsesLocal.py) can be used in place of [responses](./src/responses.py), which switches from using the Groq API to,
at the moment, llama3.1 using Ollama.

If you have an appropriate GPU, this may be preferable, but I have found the responses to be much worse on my machine.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to get involved. Make sure to follow the contribution guidelines.

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE.txt) file for more details.

