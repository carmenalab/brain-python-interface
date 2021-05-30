//
// PlexNetClient.h
//
//    Header file for PlexNetClient.cpp. See RawSocket.cpp 
//    for a console mode application that demonstrates the use 
//    of PlexNetClient to receive PlexNet data from an OmniPlex 
//    or MAP system. For questions or support, please contact
//    support@plexon.com.
//
// Author: Alex Kirillov
// (c) 1998-2014 Plexon Inc (www.plexon.com)
//

#pragma once
#include <iostream>
#include <vector>

#define PL_SingleWFType         (1)
#define PL_StereotrodeWFType    (2)     // reserved
#define PL_TetrodeWFType        (3)     // reserved
#define PL_ExtEventType         (4)
#define PL_ADDataType           (5)
#define PL_StrobedExtChannel    (257)
#define PL_StartExtChannel      (258)   // delineates frames, sent for resume also
#define PL_StopExtChannel       (259)   // delineates frames, sent for pause also
#define PL_Pause                (260)   // not used
#define PL_Resume               (261)   // not used

#define MAX_WF_LENGTH           (56)
#define MAX_WF_LENGTH_LONG      (120)

struct PL_DataBlockHeader {
    short   Type;                       // Data type; 1=spike, 4=Event, 5=continuous
    unsigned short   UpperByteOf5ByteTimestamp; // Upper 8 bits of the 40 bit timestamp
    unsigned int    TimeStamp;                 // Lower 32 bits of the 40 bit timestamp
    short   Channel;                    // Channel number
    short   Unit;                       // Sorted unit number; 0=unsorted
    short   NumberOfWaveforms;          // Number of waveforms in the data to folow, usually 0 or 1
    short   NumberOfWordsInWaveform;    // Number of samples per waveform in the data to follow
}; // size should be 16 bytes

struct PL_WaveLong {
    char    Type;                       // PL_SingleWFType, PL_ExtEventType or PL_ADDataType
    char    NumberOfBlocksInRecord;     // reserved
    char    BlockNumberInRecord;        // reserved
    unsigned char    UpperTS;           // Upper 8 bits of the 40-bit timestamp
    unsigned int    TimeStamp;          // Lower 32 bits of the 40-bit timestamp
    short   Channel;                    // Channel that this came from, or Event number
    short   Unit;                       // Unit classification, or Event strobe value
    char    DataType;                   // reserved
    char    NumberOfBlocksPerWaveform;  // reserved
    char    BlockNumberForWaveform;     // reserved
    char    NumberOfDataWords;          // number of shorts (2-byte integers) that follow this header
    short   WaveForm[MAX_WF_LENGTH_LONG];   // The actual long waveform data
}; // size should be 256 bytes


// PlexNet commands
#define PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT (10000)
#define PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_DISCONNECT_CLIENT (10999)
#define PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF (10100)
#define PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP (10200)
#define PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_STOP_DATA_PUMP (10300)
#define PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS (10400)
#define PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS (10401)

#define PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_MMF_SIZES (10001)
#define PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA (20003)
#define PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_DATA (1)

#define	PACKETSIZE	(512)

#define SPIKE_CHAN_SORTED_TIMESTAMPS (0x01)
#define SPIKE_CHAN_SORTED_WAVEFORMS (0x02)
#define SPIKE_CHAN_UNSORTED_TIMESTAMPS (0x04)
#define SPIKE_CHAN_UNSORTED_WAVEFORMS (0x08)

// header used in sending buffers that are longer than a single 512-byte buffer
struct PlexNetMultiMessageTransmissionHeader {
    int CommandCode;
    int MessageIndex; // 0-based index of multi-message transmission
    int NumberOfMessages; // num. messages in this transmission
    int NumberOfBytesInThisMessage; // valid bytes that follow this header
    int PositionOfTheFirstByte; // i.e. we are sending orig_buffer + position
};


#ifdef WIN32
#include <winsock.h>
#define socklen_t int
#else
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <string.h>
#define SOCKET int
#define INVALID_SOCKET (-1)
#define SOCKADDR sockaddr
#define SOCKADDR_IN sockaddr_in
#endif


class PlexNetClient
{
public:
    enum Protocol {
        ProtocolTCPIP,
        ProtocolUDP
    };
    PlexNetClient( const char* hostAddress, int port, Protocol theProtocol );
    virtual ~PlexNetClient( void );

    void InitSocketLibrary();
    void CreateSocket();
    void ConnectSocket();
    void SendCommandConnectClient();
    void GetPlexNetInfo();
    void ReceivePacket();
    void SelectSpikeChannels( const std::vector<unsigned char>& channelSelection );
    void SelectContinuousChannels( const std::vector<unsigned char>& channelSelection );
    void StartDataPump();

    int* ReceiveBufferIntPointer() const { return m_ReceiveBufferIntPointer; }
    bool SupportsSelectSpikeChannels() const { return m_SupportsSelectSpikeChannels; }
    bool SupportsSelectContChannels() const { return m_SupportsSelectContChannels; }
    int NumSpikeChannels() const { return m_NumSpikeChannels; }
    int NumContinuousChannels() const { return m_NumContinuousChannels; }
    const std::vector<char>& GetReceiveBuffer() const { return m_ReceiveBuffer; }

protected:
    void SendPacket();
    void ProcessMmfSizesMessage();
    void RequestParameterMMF();
    void GetNumbersOfChannels();
    void SendLongBuffer( int command, const unsigned char* buffer, int length );

    std::string m_HostAddress;
    int m_HostPort;
    Protocol m_Protocol;
    SOCKET m_Socket;
    SOCKADDR_IN m_SocketAddressIn;
    SOCKADDR m_SocketAddress;

    std::vector<char> m_SendBuffer;
    int* m_SendBufferIntPointer;
    std::vector<char> m_ReceiveBuffer;
    int* m_ReceiveBufferIntPointer;

    bool m_SupportsSelectSpikeChannels;
    bool m_SupportsSelectContChannels;
    int m_NumSpikeChannels;
    int m_NumContinuousChannels;

};

