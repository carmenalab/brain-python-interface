"""
Author: Brett Dowden
Blackrock Microsystems

Created: 4 June 2013

Define Classes to read and manipulate Cerebus Data files
"""
import Tkinter, tkFileDialog      # These are for the open file dialog
from collections import OrderedDict, namedtuple
import sys, os, mmap, struct
from datetime import datetime
import numpy as np


def readPacket(fileno, packetFields):
    """
    Read a packet from a binary data file and return a list of fields
    The amount and format of data read will be specified by the
    packetFields container
    """
    
    # First I pull out all the format strings from the basicHeaderFields
    # named tuple, then concatenate them into a string with '<' at the front (for little endian format)
    packet_formatStr = '<'+''.join([fmt for name,fmt,fun in packetFields])
    
    # Calculate how many bytes to read based on the format strings of the header fields
    bytesInPacket = struct.calcsize(packet_formatStr)
    
    packet_binary = fileno.read(bytesInPacket)
    
    # unpack the binary data from the header based on the format strings of each field.
    # This returns a list of data, but it's not always correctly formatted (eg, FileSpec
    # is read as ints 2 and 3 but I want it as '2.3'
    packet_unpacked = struct.unpack(packet_formatStr, packet_binary)    
    
    # Create a interator from the data list.  This allows a formatting function
    # to use more than one item from the list if needed, and the next formatting
    # function can pick up on the correct item in the list
    data_iter = iter(packet_unpacked)
    
    # create an empty dictionary from the name field of the packetFields.
    # The loop below will fill in the values with formatted data by calling
    # each field's formatting function
    packet_formatted = dict.fromkeys([name for name,fmt,fun in packetFields])
    for name, fmt, fun in packetFields:
        packet_formatted[name] = fun(data_iter)
    
    return packet_formatted 

            
        
def formatFileSpec(headerList):
    return str(headerList.next()) + '.' + str(headerList.next())  # eg '2.3'

def formatTimeOrigin(headerList):
    
    year        = headerList.next()
    month       = headerList.next()
    day         = headerList.next()
    dayOfWeek   = headerList.next()
    hour        = headerList.next()
    minute      = headerList.next()
    second      = headerList.next()
    millisecond = headerList.next()
             
    return datetime(year, month, day, hour, minute, second, millisecond*1000)   

def stripString(headerList):
    string = headerList.next()
    return string.split('\x00',1)[0]
             
def noFormat(headerList):
    return headerList.next()


class nsxReader():
    
    # Define a named tuple that has information about header/packet fields
    #        name - string
    #   formatStr - is the format used by struct.unpack
    #   formatFnc - is a function that formats the results of struct.unpack
    PacketDef = namedtuple('PacketDef',['name','formatStr','formatFnc'])
    
    # Define the names, data size, and format of the header fields
    basicHeaderFields = [         
        PacketDef('FileTypeID'     , '8s'  , noFormat), # 8 bytes   - 8 char array
        PacketDef('FileSpec'       , '2B'  , formatFileSpec), # 2 bytes   - 2 unsigned char
        PacketDef('BytesInHeader'  , 'I'   , noFormat), # 4 bytes   - uint32
        PacketDef('Label'          , '16s' , stripString), # 16 bytes  - 16 char array
        PacketDef('Comment'        , '256s', stripString), # 256 bytes - 256 char array
        PacketDef('Period'         , 'I'   , noFormat), # 4 bytes   - uint32
        PacketDef('TimeResolution' , 'I'   , noFormat), # 4 bytes   - uint32
        PacketDef('TimeOrigin'     , '8H'  , formatTimeOrigin), # 16 bytes  - 8 uint16
        PacketDef('ChannelCount'   , 'I'   , noFormat)  # 4 bytes   - uint32
        ]
    
    extendedHeaderFields = [         
        PacketDef('Type'             , '2s' , noFormat), # 2 bytes  - 2 char array
        PacketDef('ElectrodeID'      , 'H'  , noFormat), # 2 bytes  - uint16
        PacketDef('ElectrodeLabel'   , '16s', stripString), # 16 bytes - 16 char array
        PacketDef('PhysicalConnector', 'B'  , noFormat), # 1 byte   - uint8
        PacketDef('ConnectorPin'     , 'B'  , noFormat), # 1 byte   - uint8
        PacketDef('MinDigitalValue'  , 'h'  , noFormat), # 2 bytes   - int16
        PacketDef('MaxDigitalValue'  , 'h'  , noFormat), # 2 bytes   - int16
        PacketDef('MinAnalogValue'   , 'h'  , noFormat), # 2 bytes   - int16
        PacketDef('MaxAnalogValue'   , 'h'  , noFormat), # 2 bytes   - int16
        PacketDef('Units'            , '16s', stripString), # 16 bytes - 16 char array
        PacketDef('HighFreqCorner'   , 'I'  , lambda x: str(float(x.next())/1000)+' Hz'), # 4 bytes  - uint32
        PacketDef('HighFreqOrder'    , 'I'  , noFormat), # 4 bytes  - uint32
        PacketDef('HighFreqType'     , 'H'  , noFormat), # 2 bytes  - uint16
        PacketDef('LowFreqCorner'    , 'I'  , lambda x: str(float(x.next())/1000)+' Hz'), # 4 bytes  - uint32
        PacketDef('LowFreqOrder'     , 'I'  , noFormat), # 4 bytes  - uint32
        PacketDef('LowFreqType'      , 'H'  , noFormat)  # 2 bytes  - uint16
        ]
    
    dataHeaderFields = [
        PacketDef('Header'           , 'B' , noFormat),
        PacketDef('Timestamp'        , 'I' , noFormat),
        PacketDef('NumDataPoints'    , 'I' , noFormat)                
        ]
    
    def __init__(self, fileName=''):
        
        if not fileName:
            root = Tkinter.Tk()
            root.withdraw()
            
            self.fileName = tkFileDialog.askopenfilename()
        else:
            self.fileName = fileName

        self.fileNo = open(self.fileName, 'rb')
            
    def readHeaders(self):
        
        extendedHeader_formatted = []

        # format the header by passing the unpacked data and field information
        # to the generic packet formatter
        
        basicHeader_formatted = readPacket(self.fileNo, self.basicHeaderFields)
        
        #nsFileSize = os.path.getsize(nsFile)
        #self.data = mmap.mmap(nsFile.fileno(), nsFileSize, access=mmap.ACCESS_READ)

        
        numExtendedHeaders = basicHeader_formatted['ChannelCount']
        
        for i in range(numExtendedHeaders):
            extendedHeader_formatted.append( readPacket(self.fileNo, self.extendedHeaderFields) )
        
        self.basicHeader    = basicHeader_formatted
        self.extendedHeader = extendedHeader_formatted
     
    def readDataHeader(self):
        
        # Read the data header
        dataHeader_formatted = readPacket(self.fileNo, self.dataHeaderFields)
        
        # Check to make sure the header is valid (ie Header field = 0)
        if dataHeader_formatted['Header'] == 0:
            print "Invalid Header.  File may be corrupt"
        
        self.dataHeader = dataHeader_formatted
        
    def readData(self):
        
        shape = (self.dataHeader['NumDataPoints'], self.basicHeader['ChannelCount'])
        
        self.data = np.fromfile(file=self.fileNo, dtype=np.int16 ).reshape(shape)
        
def main():
    
    if len(sys.argv) == 1:
        ns = nsxReader()
    else:
        ns = nsxReader(sys.argv[1])
        
    ns.readHeaders()
    ns.readDataHeader()
    ns.readData()
    
    print ns.fileName
    print ns.basicHeader
    print ns.dataHeader
    print ns.data

if __name__ == '__main__':
    main()




