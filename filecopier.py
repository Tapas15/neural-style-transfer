import paramiko

# Define the SFTP server information
host = '10.10.10.71'
port = 22
username = 'simadmin'
password = '12345678'

# Remote and local file paths
local_files = [
    'Research_paper/nn.svg',

] 
remote_directory = '/home/simadmin/CA/research/'  # Remote directory

# Create an SFTP client
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Connect to the server
    client.connect(host, port, username, password)
    sftp = client.open_sftp()

    for local_file in local_files:
        remote_file = remote_directory + local_file.split('/')[-1]  # Extract file name from local path
        print(f"Copying {local_file} to {remote_file}")

        # Upload the file
        try:
            sftp.put(local_file, remote_file)
        except Exception as e:
            print(f"Error uploading {local_file}: {e}")

    # Close the SFTP session
    sftp.close()

finally:
    # Close the SSH connection
    client.close()
