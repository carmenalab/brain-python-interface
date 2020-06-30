from NatNetClient import NatNetClient

# This will create a new NatNet client
streamingClient = NatNetClient()

streamingClient.dataSocket = streamingClient.__createDataSocket(streamingClient.dataPort)
if (streamingClient.dataSocket is None):
    print("Could not open data channel")
    exit

# Create the command socket
streamingClient.commandSocket = streamingClient.__createCommandSocket()
if (streamingClient.commandSocket is None):
    print("Could not open command channel")
    exit


# receive some data

data, addr = streamingClient.dataSocket.recvfrom(1024)  # 32k byte buffer size
if (len(data) > 0):
    streamingClient__processMessage(data)
