import requests


def is_remote_server_reachable(url="http://localhost:11434/api/tags", timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        # print(f"Error reaching server: {e}")
        return False