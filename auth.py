# auth.py

# Hardcoded credentials for demonstration purposes
VALID_USERNAME = 'Parul'
VALID_PASSWORD = 'pass123' 

def authenticate(username, password):
    return username == VALID_USERNAME and password == VALID_PASSWORD
