from typing import Final
import os
from dotenv import load_dotenv
from discord import DMChannel, Intents, Client, Message
from responses import get_response

#load token
load_dotenv()
TOKEN: Final[str] = os.getenv('DISCORD_TOKEN')

#bot setup
intents: Intents = Intents.default()
intents.message_content = True
client: Client = Client(intents=intents)

#message functionality
async def send_message(message: Message, user_message: str, context_string: str, username: str) -> None:
    if not user_message:
        print('(intents are not working)')
        return
    
    if is_private := user_message[0] == '?':
        user_message = user_message[1:]
        
    try:
        response: str = get_response(user_message, context_string, username)
        await message.author.send(response) if is_private else await message.channel.send(response)
    except Exception as e:
        print(e)

#bot startup
@client.event
async def on_ready() -> None:
    print(f'{client.user} is now running.')
    
#step idk handling incoming messages
@client.event
async def on_message(message: Message) -> None:
    if message.author == client.user:
        return

    # Check if the bot is mentioned in the message or if they are in a DM
    if (client.user in message.mentions) or (isinstance(message.channel, DMChannel)):
        username: str = str(message.author)
        user_message: str = message.content
        
        print(f'[{message.channel}] {username}: "{user_message}"')
        
        # List to store formatted messages
        context_messages = []
        
        # Skip the current message by using a counter
        skip_current_message = True
        
        # Fetch the last 100 messages before the current one in the channel
        async for msg in message.channel.history(limit=26):  # Fetch 26 to account for current message
            if skip_current_message:
                skip_current_message = False
                continue  # Skip the first message, which is the current message
            
            # Format each message
            formatted_message = f"[{msg.created_at}] {msg.author.name}: {msg.content}"
            context_messages.append(formatted_message)
        
        # Create a single string with all the messages
        context_string = "\n".join(context_messages)
        
        await send_message(message, user_message, context_string, username)



#main
def main() -> None:
    client.run(token=TOKEN)
    
if __name__ == "__main__":
    main()    