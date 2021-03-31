//
// RawSocket.cpp
//
//    Console-mode app demonstrating how to receive PlexNet packets using either
//    TCP/IP or UDP.  WSAStartup and WSACleanup are Winsock-specific calls which
//    may need to be removed or replaced, depending on your platform.  Note that
//    a blocking socket is used for simplicity, but a non-blocking socket could be
//    used as well.
//
// Author: Larry Spence (larry@plexon.com)
// (c) 1998-2011 Plexon Inc (www.plexon.com)
//
#include <stdio.h>
#include <vector>
#include <iostream>

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
}; // 16 bytes

struct PL_WaveLong {
    char    Type;                       // PL_SingleWFType, PL_ExtEventType or PL_ADDataType
    char    NumberOfBlocksInRecord;     // reserved
    char    BlockNumberInRecord;        // reserved
    unsigned char    UpperTS;           // Upper 8 bits of the 40-bit timestamp
    unsigned int    TimeStamp;         // Lower 32 bits of the 40-bit timestamp
    short   Channel;                    // Channel that this came from, or Event number
    short   Unit;                       // Unit classification, or Event strobe value
    char    DataType;                   // reserved
    char    NumberOfBlocksPerWaveform;  // reserved
    char    BlockNumberForWaveform;     // reserved
    char    NumberOfDataWords;          // number of shorts (2-byte integers) that follow this header
    short   WaveForm[MAX_WF_LENGTH_LONG];   // The actual long waveform data
}; // size should be 256


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

void ProcessBuffer( char* buf );
using namespace std;

#ifdef WIN32
#include <winsock.h>
#define socklen_t int

bool InitWindowsSockets()
{
    WSADATA   wsaData;
    WORD wVersionRequested = MAKEWORD( 2, 2 );
    int err = WSAStartup( wVersionRequested, &wsaData );
    if ( err != 0 ) {
        cout << "ERROR: Cannot init Winsock" << endl;
        return false;
    } else {
        cout << "Winsock initialized" << endl;
        return true;
    }
}
void Close( SOCKET s )
{
    if ( s != INVALID_SOCKET ) {
        closesocket( s );
    }
    WSACleanup();
}
#else
#include <iostream>
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

bool InitWindowsSockets()
{
    return true;
}

void Close( SOCKET s )
{
    if ( s != INVALID_SOCKET ) {
        close( s );
    }
}
#endif

// IP address of the remote machine to which the MAP is connected;
// substitute the actual IP address of your machine
#define HOSTADDRESS "192.168.37.106"

// Port used by PlexNet on the machine to which the MAP is connected.
// It can be set in the PlexNetLocal options dialog.
#define PLEXNET_PORT (6000)

// TCP/IP or UDP protocol; PlexNetLocal options must be set accordingly.
// Note that PlexNet using UDP only supports a single remote client such
// as this one, whereas TCP/IP allows multiple remote clients.
// Important: UDP does not guarantee packet arrival and should only
// be used with a high-reliability network such as a LAN.
#define PLEXNET_UDP (0)
#define PLEXNET_TCP (1)
#define PLEXNET_PROTOCOL PLEXNET_TCP

bool SendBuffer( const bool bUseTCP, SOCKET s, char * SendBuf, SOCKADDR_IN& si )
{
    int res = 0;
    int bytes = 0;
    int attempt = 0;
    while ( bytes < PACKETSIZE && attempt < 1000 ) {
        attempt++;
        if ( bUseTCP ) {
            res = send( s, SendBuf + bytes, PACKETSIZE - bytes, 0 );
        } else { // UDP
            res = sendto( s, SendBuf + bytes, PACKETSIZE - bytes, 0, ( SOCKADDR* )&si, sizeof( si ) );
        }
        bytes += res;
    }
    if ( bytes != PACKETSIZE ) {
        cout << "ERROR: Cannot send" << endl;
        Close( s );
        return false;
    }
    return true;
}

bool ReceivePacket( const bool bUseTCP, SOCKET s, char * ReceiveBuf )
{
    SOCKADDR_IN SenderAddr; // for UDP recvfrom only
    socklen_t SenderAddrSize = sizeof( SenderAddr );
    int res = 0;
    int bytes = 0;
    int attempt = 0;
    while ( bytes < PACKETSIZE && attempt < 1000 ) {
        attempt++;
        if ( bUseTCP ) {
            res = recv( s, ReceiveBuf + bytes, PACKETSIZE - bytes, 0 );
        } else { // UDP
            res = recvfrom( s, ReceiveBuf + bytes, PACKETSIZE - bytes, 0, ( SOCKADDR* )&SenderAddr, &SenderAddrSize );
        }
        bytes += res;
    }
    if ( bytes != PACKETSIZE ) {
        cout << "ERROR: Cannot receive: bytes=" << bytes << " attempts=" << attempt << endl ;
        Close( s );
        return false;
    }
    return true;
}


int main( int argc, char* argv[] )
{
    char SendBuf[1024];
    char ReceiveBuf[1024];
    int res;

    const bool bUseTCP = ( PLEXNET_PROTOCOL == PLEXNET_TCP );

    if ( !InitWindowsSockets() ) {
        return 0;
    }

    // create socket
    SOCKET s = socket( AF_INET, bUseTCP ? SOCK_STREAM : SOCK_DGRAM, 0 );
    if ( s == INVALID_SOCKET ) {
        cout << "ERROR: cannot create socket\n" << endl;
        Close( s );
        return 0;
    } else {
        cout << "Socket created" << endl;
    }

    SOCKADDR_IN si;
    memset( &si, 0, sizeof( si ) );
    //si.sin_addr.S_un.S_addr = inet_addr( HOSTADDRESS );
    si.sin_family = AF_INET;
    hostent* he = gethostbyname( HOSTADDRESS );
    si.sin_addr = *( ( in_addr* )he->h_addr );

    // port is configurable. look up port number
    // at the top of PlexNetLocal dialog

    si.sin_port = htons( PLEXNET_PORT );
    SOCKADDR sa;
    memset( &sa, 0, sizeof( sa ) );
    memcpy( &sa, &si, sizeof( sa ) );

    if ( bUseTCP ) {
        // we're ready to connect to PlexNet
        cout << "Connecting to PlexNet..." << endl;
        res = connect( s, ( SOCKADDR* )&sa, sizeof( SOCKADDR ) );
        if ( res != 0 ) {
            cout << "ERROR: cannot connect" << endl;
            Close( s );
            return 0;
        } else
            cout << "Connected to PlexNet at address " << HOSTADDRESS << ", port " <<  PLEXNET_PORT << endl;
    } else {
        // UDP
        // don't need to call connect() since UDP is connectionless
    }

    int* sendbuf = ( int* )SendBuf;
    int* recbuf = ( int* )ReceiveBuf;

    // send command to connect the client and set the data transfer mode
    memset( SendBuf, 0, PACKETSIZE );
    sendbuf[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT;

    // this command specifies general data transfer options: what data types we want to transfer
    // we will need to send additional commands later to specify details of data transfer

    // here we specify that we want everything.
    // we will send other commands later where we will specify what data we want from what channels
    sendbuf[1] = 1; // want timestamps
    sendbuf[2] = 1; // want spike waveforms
    sendbuf[3] = 1; // want analog data
    sendbuf[4] = 1;  // spike channel from
    sendbuf[5] = 128; // spike channel to

    if ( !SendBuffer( bUseTCP, s, SendBuf, si ) ) {
        return 0;
    }

    cout << "Sent transfer mode command" << endl;

    // read the reply - acknowledgment
    if ( !ReceivePacket( bUseTCP, s, ReceiveBuf ) ) {
        return 0;
    }

    cout << "Received packet, code " << recbuf[0] << endl;

    bool supportsSelectSpikeChannels = false;
    bool supportsSelectContChannels = false;

    if ( recbuf[0] == PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_MMF_SIZES ) {
        // recbuf[3] contains the number of supported commands
        int numCommands = recbuf[3];
        if ( numCommands > 0 && numCommands < 32 ) {
            for( int i = 0; i < numCommands; i++ ) {
                if ( recbuf[4 + i] == PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS ) {
                    supportsSelectSpikeChannels = true;
                }
                if ( recbuf[4 + i] == PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS ) {
                    supportsSelectContChannels = true;
                }
            }
        }
    }

    int NumSpikeChannels = 0;
    int NumContinuousChannels = 0;

    if ( supportsSelectSpikeChannels && supportsSelectContChannels ) {
        // request parameters
        memset( SendBuf, 0, PACKETSIZE );
        sendbuf[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF;
        if ( !SendBuffer( bUseTCP, s, SendBuf, si ) ) {
            return 0;
        }

        // read parameters until we get PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA
        bool gotServerArea = false;
        while ( !gotServerArea ) {
            if ( !ReceivePacket( bUseTCP, s, ReceiveBuf ) ) {
                return 0;
            }
            if ( recbuf[0] == PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA ) {
                NumSpikeChannels = recbuf[15];
                NumContinuousChannels = recbuf[17];
                gotServerArea = true;
            }
        }

        memset( SendBuf, 0, PACKETSIZE );
        sendbuf[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS;
        sendbuf[1] = 0; // Packet index!!!
        sendbuf[2] = 1; // number of packets
        sendbuf[3] = NumSpikeChannels; // Number of bytes
        sendbuf[4] = 0; // byte offset

        // specify what data we want for spike channels
        // we have 1 byte for each channel starting with byte 20
        // in each byte we can specify any combination of the following flags:
        // #define SPIKE_CHAN_SORTED_TIMESTAMPS (0x01)
        // #define SPIKE_CHAN_SORTED_WAVEFORMS (0x02)
        // #define SPIKE_CHAN_UNSORTED_TIMESTAMPS (0x04)
        // #define SPIKE_CHAN_UNSORTED_WAVEFORMS (0x08)

        // the following for loop instructs PlexNetLocal to send everything from every spike channel
        //for ( int i = 0; i < NumSpikeChannels; i++ )
        //{
        //    SendBuf[20 + i] = SPIKE_CHAN_SORTED_TIMESTAMPS | SPIKE_CHAN_SORTED_WAVEFORMS | SPIKE_CHAN_UNSORTED_TIMESTAMPS | SPIKE_CHAN_UNSORTED_WAVEFORMS;
        //}


        // here is an example on how to specify data types in mode detail:
        // example 1: send only sorted timestamps for the first spike channel:
        int spikeChannel = 1;
        SendBuf[20 + spikeChannel - 1] = SPIKE_CHAN_SORTED_TIMESTAMPS;
        // example 2: send sorted timestamps and waveforms for the second spike channel:
        spikeChannel = 2;
        SendBuf[20 + spikeChannel - 1] = SPIKE_CHAN_SORTED_TIMESTAMPS | SPIKE_CHAN_SORTED_WAVEFORMS;
        // example 3: send sorted and unsorted timestamps and waveforms for spike channel 5:
        spikeChannel = 5;
        SendBuf[20 + spikeChannel - 1] = SPIKE_CHAN_SORTED_TIMESTAMPS | SPIKE_CHAN_SORTED_WAVEFORMS | SPIKE_CHAN_UNSORTED_TIMESTAMPS | SPIKE_CHAN_UNSORTED_WAVEFORMS;

        if ( !SendBuffer( bUseTCP, s, SendBuf, si ) ) {
            return 0;
        }

        memset( SendBuf, 0, PACKETSIZE );
        sendbuf[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS;
        sendbuf[1] = 0;
        sendbuf[2] = 1;
        sendbuf[3] = NumContinuousChannels;
        sendbuf[4] = 0;

        // specify what continuous channels we want to receive
        // to get the list of continuous channels,
        // please run PlexNetRemote on the same computer as PlexNexLocal, connect,
        // press Data Transfer Options and tab to Continuous Channels tab
        // channels are listed with 1-based channel numbers
        
        // the following for loop instructs PlexNetLocal to send all continuous channels
        //for ( int i = 0; i < NumContinuousChannels; i++ )
        //{
        //    SendBuf[20 + i] = 1;
        //}

        // example 1: get only channels 1, 2, and 17
        int contChannel = 1;
        SendBuf[20 + contChannel - 1 ] = 1;
        contChannel = 2;
        SendBuf[20 + contChannel - 1 ] = 1;
        contChannel = 17;
        SendBuf[20 + contChannel - 1 ] = 1;

        if ( !SendBuffer( bUseTCP, s, SendBuf, si ) ) {
            return 0;
        }
    }

    // start data pump
    cout << "Start data pump..." << endl;
    memset( SendBuf, 0, PACKETSIZE );
    sendbuf[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP;
    if ( !SendBuffer( bUseTCP, s, SendBuf, si ) ) {
        return 0;
    }
    cout << "Started data pump" << endl;

    int numPackets = 128;
    // receive and process numPackets buffers
    // typically, a real app would sit in a loop receiving and processing packets as long as desired
    cout << "Waiting for data..." << endl;
    SOCKADDR_IN SenderAddr; // for UDP recvfrom only
    socklen_t SenderAddrSize = sizeof( SenderAddr );
    for ( int i = 0; i < numPackets; i++ ) {
        if ( !ReceivePacket( bUseTCP, s, ReceiveBuf ) ) {
            return 0;
        }

        if ( recbuf[0] == 1 ) { // we have a buffer to process
            ProcessBuffer( ReceiveBuf );
        }
    }

    // stop data pump
    cout << "Stopping data pump..." << endl;
    memset( SendBuf, 0, PACKETSIZE );
    sendbuf[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_STOP_DATA_PUMP;
    if ( !SendBuffer( bUseTCP, s, SendBuf, si ) ) {
        return 0;
    }
    cout << "Stopped data pump" << endl;

    // disconnect
    cout << "Disconnecting..." << endl;
    memset( SendBuf, 0, PACKETSIZE );
    sendbuf[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_DISCONNECT_CLIENT;
    if ( !SendBuffer( bUseTCP, s, SendBuf, si ) ) {
        return 0;
    }
    cout << "Disconnected" << endl;

    // we're done
    Close( s );
    return 1;
}


void ProcessBuffer( char* m_RecBuf )
{
    int* ibuf = ( int* )m_RecBuf;

    // decode the buffer

    int NumServerDropped = ibuf[2];
    int NumMMFDropped = ibuf[3];

    PL_DataBlockHeader db;
    PL_WaveLong wave;
    int sdb = sizeof( db );
    int sw = sizeof( PL_WaveLong );

    int nbuf;
    short buf[128];
    int nwaves = 0;
    int nts = 0;
    int found_eop = 0;

    int length = PACKETSIZE - 16; // type and msg count
    int pos = 16;

    while ( pos + sdb <= PACKETSIZE ) {
        // extract one PL_DataBlockHeader
        memcpy( &db, m_RecBuf + pos, sdb );

        if ( db.Type == 0 ) {// empty block
            break;
        }

        if ( db.Type == -1 ) { // end of packet, we're done
            found_eop = 1;
            break;
        }

        pos += sdb;
        nbuf = 0;

        if ( db.NumberOfWaveforms > 0 ) {
            // get the waveform values
            nbuf = db.NumberOfWaveforms * db.NumberOfWordsInWaveform;
            memcpy( buf, m_RecBuf + pos, nbuf * 2 );
            pos += nbuf * 2;
        }

        // fill in a PL_WaveLong from the packet
        memset( &wave, 0, sw );
        wave.Type = ( char )db.Type;
        wave.Channel = db.Channel;
        wave.Unit = db.Unit;
        wave.TimeStamp = db.TimeStamp;
        wave.UpperTS = ( char )db.UpperByteOf5ByteTimestamp;
        wave.NumberOfDataWords = nbuf;
        if ( nbuf > 0 ) {
            memcpy( wave.WaveForm, buf, nbuf * 2 );
        }
        if ( wave.Type == PL_SingleWFType ) { // spike timestamp, with or without waveform values
            if ( nbuf > 0 )	{ // it has waveform values
                nwaves++;
            } else { // no waveform values, timestamp only
                nts++;
            }
        }
        if ( wave.Type == PL_ExtEventType ) {
            nts++;
        }
        cout << "type= " << ( int )wave.Type << " chan=" << ( int )wave.Channel << " unit=" << ( int )wave.Unit << " t=" << wave.TimeStamp << endl;
    }
}
