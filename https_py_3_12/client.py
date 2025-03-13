import os 
import requests

def get_secret_message():
    url = os.getenv("SERVER_URL")
    response = requests.get(url)
    print("Response from server: ", response.text)

if __name__ == "__main__":
    get_secret_message()