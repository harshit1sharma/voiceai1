# auth.py
#final test
# Hardcoded credentials for demonstration purposes
VALID_USERNAME = 'ayush'
VALID_PASSWORD = 'pass123' 

def authenticate(username, password):
    return username == VALID_USERNAME and password == VALID_PASSWORD
