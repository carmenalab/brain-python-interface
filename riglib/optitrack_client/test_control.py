from NatNetClient import NatNetClient 


# This is a callback function that gets connected to the NatNet client and called once per mocap frame.
def receiveNewFrame( frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
    #print( "Received frame", frameNumber )
    pass

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame( id, position, rotation ):
    #print( "Received frame for rigid body", position )
    pass

# This will create a new NatNet client
test_client = NatNetClient()

# Configure the streaming client to call our rigid body handler on the emulator to send data out.
test_client.newFrameListener = receiveNewFrame
test_client.rigidBodyListener = receiveRigidBodyFrame

test_client.sendCommand( test_client.NAT_REQUEST_MODELDEF, "", test_client.commandSocket, 
                (test_client.serverIPAddress, test_client.commandPort) )