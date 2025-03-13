from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher_suite = Fernet(key)
cipher_msg = cipher_suite.encrypt(b"A really secret message. Not for prying eyes.")
print(cipher_msg)
