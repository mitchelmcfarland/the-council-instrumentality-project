import os
import json

def get_all_content_from_json_files(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
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
                                    outfile.write(f"{username} : {content} : {timestamp}\n")
                    except json.JSONDecodeError:
                        print(f"Error parsing {filename}")

directory_path = '.'  
output_file = 'timestamped_fixed.txt'
get_all_content_from_json_files(directory_path, output_file)
print(f"Messages saved to {output_file}")