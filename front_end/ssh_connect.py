import paramiko


def connect(hostname, username, key_file, directory):
    try:
        # Create an SSH client
        client = paramiko.SSHClient()

        # Automatically add the server's host key
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        private_key = paramiko.RSAKey.from_private_key_file(key_file)

        # Connect to the SSH server (default port 22)
        client.connect(hostname, username=username, pkey=private_key)

        # Source the virtual environment and change directory
        command = f'source .myvenv/.rag/bin/activate && cd {directory} && python3 test.py'
        stdin, stdout, stderr = client.exec_command(command)

        # Read command output to ensure it executed correctly
        output = stdout.read().decode()
        errors = stderr.read().decode()

        if output:
            print("Initial command output:\n", output)
        if errors:
            print("Initial command errors:\n", errors)

        return client
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def run(client, command):
    try:
        stdin, stdout, stderr = client.exec_command(command)

        # Read command output and errors
        output = stdout.read().decode()
        errors = stderr.read().decode()

        if output:
            print("Output:\n", output)
        if errors:
            print("Errors:\n", errors)
    except Exception as e:
        print(f"An error occurred: {e}")

    print(output)
    return output


def ssh_connect(hostname, username, key_file, directory):
    try:
        # Create an SSH client
        client = paramiko.SSHClient()

        # Automatically add the server's host key
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        private_key = paramiko.RSAKey.from_private_key_file(key_file)

        # Connect to the SSH server (default port 22)
        client.connect(hostname, username=username, pkey=private_key)

        # Execute a command to change directory and run the Python script
        command = f'source .myvenv/.rag/bin/activate && {directory} && python3 test.py'
        stdin, stdout, stderr = client.exec_command(command)

        # Read command output
        output = stdout.read().decode()
        # errors = stderr.read().decode()

        if output:
            print("Output:\n", output)
        # if errors:
        #     print("Errors:\n", errors)

        # Close the connection
        client.close()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Replace with your server details
    client = connect('163.187.168.192', 'PZheng',
                     '/home/hermes/.ssh/server', "coding/rag")
    output = run(client, "ls -a")
    print(output)
    client.close()
