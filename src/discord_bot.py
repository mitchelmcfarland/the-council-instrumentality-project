import os
from dotenv import load_dotenv
import discord

import llamacpp

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if (isinstance(message.channel, discord.DMChannel)) or client.user in message.mentions:
        await message.channel.typing()
        response = await llamacpp.get_ai_response(message.content)
        if len(response) > 2000:
            await message.channel.send("Too long!")
            return
        await message.channel.send(response)

client.run(TOKEN)
