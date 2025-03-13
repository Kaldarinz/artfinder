import os 
from cryptography.fernet import Fernet
import requests

def get_secret_message():
    url = os.getenv("SERVER_URL")
    key = os.getenv("SECRET_KEY")
    my_cipher = Fernet(key)
    response = requests.get(url)
    plain_response = my_cipher.decrypt(response.content)
    print("Response from server: ", plain_response)

if __name__ == "__main__":
    get_secret_message()