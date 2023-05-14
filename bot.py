import discord
from predict import predict

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

    # get the channel object by ID
    channel = client.get_channel(1107329292699963462)

    # create an embed object
    embed = discord.Embed(title='AI Training', description='Result of AI generated score for Toxic', color=discord.Color.blue())

    # set fields for the embed
    embed.add_field(name='Message', value=message.content, inline=False)
    
    prediction = predict(message.content)
    
    result = ""
    violations = [key for key, value in prediction.items() if value > 0.8]
    result += ", ".join(violations)
    embed.add_field(name='Score', value=prediction, inline=True)
    await channel.send(embed=embed)
    if len(result)>0:
        await message.reply(f"Your message might have a word that breaks the rules as it is marked as {result} by our AI")

client.run('')

