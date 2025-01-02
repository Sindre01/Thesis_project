import subprocess
import time
import requests

# Configuration
SSH_ALIAS = "fox"  # SSH alias from ~/.ssh/config
LOCAL_PORT = 11434  # Local port for SSH tunnel
REMOTE_PORT = 11434 # Remote port where Ollama server is running
OLLAMA_URL = f"http://localhost:{LOCAL_PORT}"  # Ollama API URL


def check_ssh_session():
    """Check if an SSH session is active."""
    try:
        result = subprocess.run([
            "ssh", "-O", "check", SSH_ALIAS
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if "Master running" in result.stdout.decode():
            return True
    except Exception as e:
        print(f"Error checking SSH session: {e}")
    return False


def is_port_forwarding_active():
    """Check if port forwarding is active on the specified port."""
    try:
        result = subprocess.run([
            "lsof", "-i", f"tcp:{LOCAL_PORT}"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return bool(result.stdout)
    except Exception as e:
        print(f"Error checking port forwarding: {e}")
        return False


def start_ollama_server():
    """Start the Ollama server on the remote cluster."""
    try:
        print("Starting Ollama server on remote cluster...")
        subprocess.run([
            "ssh", SSH_ALIAS, "ollama serve &"
        ])
        print("Ollama server started on remote cluster.")
    except Exception as e:
        print(f"Failed to start Ollama server: {e}")



def start_ssh_forwarding_tunnel():
    """Start SSH tunnel if not already active."""
    if not check_ssh_session() and not is_port_forwarding_active():
        print("Starting SSH tunnel...")
        try:
            subprocess.run([
                "ssh", "-f", "-N", "-L",
                f"{LOCAL_PORT}:localhost:{REMOTE_PORT}", SSH_ALIAS
            ])
            print("SSH tunnel started.")
        except Exception as e:
            print(f"Failed to start SSH tunnel: {e}")
            exit(1)
    elif is_port_forwarding_active():
        print("Port forwarding is already active.")
    else:
        print("SSH tunnel is already active.")


def test_ollama_connection_and_forwarding():
    """Test the Ollama server connection."""
    try:
        response = requests.get(OLLAMA_URL)
        if response.status_code == 200:
            print("Ollama is running")
        else:
            print(f"Unexpected response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")


def close_ssh_session():
    """Close the active SSH session."""
    try:
        subprocess.run([
            "ssh", "-O", "exit", SSH_ALIAS
        ])
        print("SSH session closed.")
    except Exception as e:
        print(f"Failed to close SSH session: {e}")


if __name__ == "__main__":

    # Start SSH tunnel and forwarding
    start_ssh_forwarding_tunnel()

    # Start Ollama server on the remote cluster
    start_ollama_server()

    # Wait briefly for tunnel setup
    time.sleep(2)

    # Test Ollama server connection to the fox cluster
    test_ollama_connection_and_forwarding()

    # Uncomment the following line if you want to close the SSH session automatically
    # close_ssh_session()
