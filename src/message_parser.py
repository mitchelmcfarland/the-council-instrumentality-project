import os
import json
from datetime import datetime
import ast
from dotenv import load_dotenv
import os

dotenv.load_dotenv('.env')
username_dict = ast.literal_eval(str(os.getenv("USERNAME_DICT")))
#USERNAME_DICT should be in the .env file as a string of format USERNAME_DICT = "{'username',: 'realname', ...}

def get_all_content_from_json_files(directory, output_file):
    messages = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for message in data:
                            if "content" in message and "author" in message and "timestamp" in message:
                                content = message["content"]
                                username = message["author"].get("username", "Unknown")
                                timestamp = message["timestamp"]
                                real_name = username_dict.get(username, username)
                                try:
                                    parsed_timestamp = datetime.fromisoformat(timestamp)
                                    messages.append((real_name, content, timestamp))
                                except ValueError:
                                    print(f"Skipping message with invalid timestamp: {timestamp}")
                except json.JSONDecodeError:
                    print(f"Error parsing {filename}")

    if messages:
        messages.sort(key=lambda x: datetime.fromisoformat(x[2]), reverse=True)
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for real_name, content, timestamp in messages:
                outfile.write(f"{real_name} : {content} : {timestamp}\n")
        print(f"Messages saved to {output_file}")
    else:
        print("No valid messages found.")

directory_path = '.'
output_file = 'timestamped_messages.txt'
get_all_content_from_json_files(directory_path, output_file)
