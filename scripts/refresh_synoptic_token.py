import requests
import re
import os
import sys

def refresh_token():
    """
    Fetches the latest Synoptic API token from the public weather.gov source
    and updates the .env file.
    """
    url = "https://www.weather.gov/source/wrh/apiKey.js"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print(f"Fetching token from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content = response.text
        
        # Look for var mesoToken='...';
        match = re.search(r"var\s+mesoToken\s*=\s*['\"]([^'\"]+)['\"]", content)
        if not match:
            print("Error: Could not find mesoToken in the response.")
            sys.exit(1)
            
        token = match.group(1)
        print(f"Found token: {token}")
        
        # Update .env file
        env_path = ".env"
        if not os.path.exists(env_path):
            print(f"Warning: {env_path} not found. Creating a new one.")
            with open(env_path, "w") as f:
                f.write(f"SYNOPTIC_TOKEN={token}\n")
        else:
            with open(env_path, "r") as f:
                lines = f.readlines()
            
            updated = False
            new_lines = []
            for line in lines:
                if line.startswith("SYNOPTIC_TOKEN="):
                    new_lines.append(f"SYNOPTIC_TOKEN={token}\n")
                    updated = True
                else:
                    new_lines.append(line)
            
            if not updated:
                new_lines.append(f"\nSYNOPTIC_TOKEN={token}\n")
            
            with open(env_path, "w") as f:
                f.writelines(new_lines)
                
        print(f"Successfully updated {env_path} with the latest token.")
        return token

    except Exception as e:
        print(f"Failed to refresh token: {e}")
        sys.exit(1)

if __name__ == "__main__":
    refresh_token()
