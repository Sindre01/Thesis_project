import sys
import pexpect

def run_bash_script_with_ssh(path):
    # Define the SSH command to execute the script remotely
    ssh_command = f"bash {path}"

    try:
        # Start the script using pexpect to handle 2FA
        process = pexpect.spawn(ssh_command, timeout=30)

        # Decode bytes to str and write logs to stdout
        process.logfile = sys.stdout.buffer  # Write raw bytes instead

        # Wait for password prompt
        process.expect("password:")
        process.sendline("your_password_here")  # Replace with your SSH password

        # Wait for 2FA prompt
        process.expect("Verification code:")
        process.sendline(input("Enter 2FA code: "))  # Dynamic input for 2FA

        # Wait for the command to complete
        process.expect(pexpect.EOF)
        output = process.before.decode("utf-8")  # Decode bytes to string

        print("SSH tunnel established successfully.")

        # Print the output
        print(output)

    except pexpect.exceptions.EOF:
        print("Connection closed unexpectedly.")
    except pexpect.exceptions.TIMEOUT:
        print("Connection timed out.")
    except Exception as e:
        print(f"An error occurred: {e}")

# # Call the function
run_bash_script_with_ssh(sys.argv[1])
