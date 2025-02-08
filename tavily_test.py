from dotenv import load_dotenv
import os
import requests

# Load .env file
load_dotenv()

# Fetch the Tavily API key
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("Tavily API key not found. Please set it in the .env file.")

# Function to query Tavily API
def query_tavily_api(query, api_key):
    url = "https://api.tavily.com/search"  # Replace with the correct Tavily API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "num_results": 5  # Adjust as needed
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()  # Assuming JSON response
    else:
        raise Exception(f"API Request failed: {response.status_code} - {response.text}")

# Test the API query
if __name__ == "__main__":
    query = "Artificial intelligence trends 2025"
    try:
        print("Querying Tavily API...")
        results = query_tavily_api(query, api_key)
        print("Tavily API Results:")
        print(results)  # Ensure something is being printed
    except Exception as e:
        print(f"Error: {e}")
