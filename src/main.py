from typing import Final
import os
from dotenv import load_dotenv
from discord import DMChannel, Intents, Client, Message
from responses import get_response

# load token
load_dotenv()
TOKEN: Final[str] = os.getenv('DISCORD_TOKEN')

# bot setup
intents: Intents = Intents.default()
intents.message_content = True
client: Client = Client(intents=intents)

# message functionality
async def send_message(message: Message, username: str, message_content: str, current_conversation: str) -> None:
    if not message_content:
        print('(intents are not working)')
        return

    try:
        response: str = get_response(message_content, current_conversation, username)
        await message.author.send(response) if isinstance(message.channel, DMChannel) else await message.channel.send(response)
    except Exception as e:
        print(e)

# bot startup
@client.event
async def on_ready() -> None:
    print(f'{client.user} is now running.')

# handling incoming messages
@client.event
async def on_message(message: Message) -> None:
    if message.author == client.user:
        return

    # Check if the bot is mentioned in the message or if they are in a DM
    if (client.user in message.mentions) or (isinstance(message.channel, DMChannel)):

        # Get the display name and cleaned message content separately
        username = message.author.display_name
        message_content = message.clean_content

        print(f'[{message.channel}] {username}: "{message_content}"')

        # List to store formatted messages for the conversation history
        context_messages = []

        # Fetch the last 100 messages before the current one in the channel
        async for msg in message.channel.history(limit=100):
            # Format each message and strip mentions using clean_content
            formatted_message = f"{msg.author.display_name}: {msg.clean_content}"
            context_messages.append(formatted_message)

            # Stop when we've collected 11 messages
            if len(context_messages) == 11:
                break

        # Create a single string with all the conversation messages
        current_conversation = "\n".join(context_messages)

        # Pass the current cleaned message content and username separately, along with the formatted conversation
        await send_message(message, username, message_content, current_conversation)

# main
def main() -> None:
    client.run(TOKEN)

if __name__ == "__main__":
    main()
