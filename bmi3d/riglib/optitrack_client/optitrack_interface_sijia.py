import numpy as np
import sys, time
import socket


N_TEST_FRAMES = 1 #number of testing frames during start
class System(object):
    """
    this is is the dataSource interface for getting the mocap at BMI3D's reqeust
    compatible with DataSourceSystem
    uses data_array to keep track of the lastest buffer
    """
    port_num  = 1230 #same as the optitrack #default to 1230
    HEADERSIZE = 10
    rece_byte_size = 512
    debug = True
    optitrack_ip_addr = "10.155.206.1"
    TIME_OUT_TIME = 2

    #a list of supported commands
    SUPPORTED_COMMANDS = [
        "start_rec",
        "send_markers",
        'send_rigid_bodies',
        'stop',
        'start'
    ]

    SUPPORTED_STREAM_TYPES = [
        'rigid_bodies',
        'markers'
        #
    ]



    rigidBodyCount = 1
    update_freq = 120 
    dtype = np.dtype((np.float, (rigidBodyCount, 6))) #6 degress of freedom
    
    def __init__(self):
        self.rigid_body_count = 1 #for now,only one rigid body


    def start(self, stream_type = "rb"):
        """
        stream_type
        """
        #start to connect to the client
                #set up the socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.settimeout(self.TIME_OUT_TIME)
  
        print("connecting to the c# server")
        '''
        self.s.bind(('', 1230)) #bind to all incoming request
        self.s.listen() #listen to one clinet
        '''
        try:
            #clientsocket, address = self.s.accept()
            self.s.connect((self.optitrack_ip_addr, self.port_num))
        except:
            print("cannot connect to Motive")
            print("Is the c# server running?")
        
        #otherwise it works as expected and set client to be a 
        #class property
        #self.clientsocket = clientsocket
        print(f"Connection to c# client \
             {self.optitrack_ip_addr} has been established.")


        if stream_type in self.SUPPORTED_STREAM_TYPES: 
            self.send_command('send_'+stream_type)
        else:
            raise Exception(f'{stream_type} is not supported \n\n suported stream types are\n{self.SUPPORTED_STREAM_TYPES}')

            

        #automatically pull 10 frames
        # and cal the mean round trip time
        t1 = time.perf_counter()
        for i in range(N_TEST_FRAMES): self.get()
        t2 = time.perf_counter()
        print(f'time to grab {N_TEST_FRAMES} frames : \
              {(t2 - t1)} s ')
                        

    def stop(self):
        msg = "stop"
        self.send_command(msg)
        #close the socket
        #self.s.close()
        print("socket closed!")
    
    def get(self):
        #the property  that gets one frame of data
        # 3 positions and 3 angles
        #the last element is frame number
        msg = "get"
        result_string = self.send_and_receive(msg)
        motive_frame = np.fromstring(result_string, sep=',')
        current_value = motive_frame[:6] #only using the motion data
        current_value.transpose()


        #for some weird reason, the string needs to be expanded..
        #just send the motion data for now
        current_value = np.expand_dims(current_value, axis = 0)
        current_value = np.expand_dims(current_value, axis = 0)
        return current_value
        


    def send_command(self, msg):
        """
        a function that sends the command to remote optitrack
        msg(string): of the message
                    the msg string has to be in the self.SUPPORTED_COMMANDS
                    
        """
        #check if the command is supported
        if msg not in self.SUPPORTED_COMMANDS: raise Exception(f'{msg} not supported')


        #get the message in string and encode in  bytes and send to the socket
        msg = f"{len(msg):<{self.HEADERSIZE}}"+msg
        msg_ascii = msg.encode("ascii")
        self.s.send(msg_ascii)

    def send_and_receive(self, msg):
        #this function sends a command
        #and then wait for a response
        msg = f"{len(msg):<{self.HEADERSIZE}}"+msg
        msg_ascii = msg.encode("ascii")
        self.s.send(msg_ascii)
        result_in_bytes = self.s.recv(self.rece_byte_size)
        return str(result_in_bytes,encoding="ASCII")
    

class Simulation(System):
    '''
    this class does all the things except when the optitrack is not broadcasting data
    the get function starts to return random numbers
    '''
    update_freq = 60 #Hz
    def get(self):
        mag_fac = 10
        current_value = np.random.rand(self.rigidBodyCount, 6) * mag_fac
        current_value = np.expand_dims(current_value, axis = 0)
        return current_value
        

if __name__ == "__main__":
    s = System()
    s.start()
    s.send_command("start_rec")
    #s.send_command("send_markers")
    time.sleep(5)
    s.stop()
    print("finished")
