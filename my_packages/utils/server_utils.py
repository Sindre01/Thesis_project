import os
import time
from subprocess import call
import requests

def is_remote_server_reachable(url="http://localhost:11434/api/tags", timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        print(f"Url: {url}")
        print(f"Response: {response}")
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        # print(f"Error reaching server: {e}")
        return False
    
def server_diagnostics(host="http://localhost:11434"):
    if is_remote_server_reachable(host + "/api/tags"):
        print("Server is reachable.")
    else:
        print("Server is not reachable. Running diagnostics...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        file_path = os.path.join(project_root, f'scripts/SSH_FORWARDING.sh')
        rc = call(file_path)
        # Check the result
        if rc == 0:
            print("Executed ssh forwarding successfully! Check job status on the server if failure continues.")
        else:
            print(f"SSH command failed with return code {rc}")
            print(f"Try to manually connect to the server again. ")
        try:
            #Wait one minute before trying again
            for remaining in range(60, 0, -1):
                print(f"Next try in {remaining} seconds", end="\r")
                time.sleep(1)  # Wait for 1 second
            print("Time's up!                          \n")  # Clear the line after completion
        except KeyboardInterrupt:
            print("\nCountdown interrupted!")