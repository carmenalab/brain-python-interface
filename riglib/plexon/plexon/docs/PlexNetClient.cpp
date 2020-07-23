//
// PlexNetClient.cpp
//
//    Implements the PlexNetClient class. See RawSocket.cpp 
//    for a console mode application that demonstrates the use 
//    of PlexNetClient to receive PlexNet data from an OmniPlex 
//    or MAP system. For questions or support, please contact
//    support@plexon.com.
//
// Author: Alex Kirillov
// (c) 1998-2014 Plexon Inc (www.plexon.com)
//
#include "PlexNetClient.h"
#include <stdexcept>

#ifdef WIN32
// this line will ensure linking with ws2_32.lib
#pragma comment( lib, "ws2_32.lib" )
#endif

using namespace std;

PlexNetClient::PlexNetClient( const char* hostAddress, int port, PlexNetClient::Protocol theProtocol )
    : m_HostAddress( hostAddress )
    , m_HostPort( port )
    , m_Protocol( theProtocol )
    , m_Socket( INVALID_SOCKET )
    , m_SupportsSelectContChannels( false )
    , m_SupportsSelectSpikeChannels( false )
    , m_NumSpikeChannels( 0 )
    , m_NumContinuousChannels( 0 )
{
    m_SendBuffer.resize( PACKETSIZE );
    m_SendBufferIntPointer = ( int* )( &m_SendBuffer[0] );
    m_ReceiveBuffer.resize( PACKETSIZE );
    m_ReceiveBufferIntPointer = ( int* )( &m_ReceiveBuffer[0] );
}


PlexNetClient::~PlexNetClient( void )
{
    if ( m_Socket != INVALID_SOCKET ) {
        // closesocket( m_Socket );
        close( m_Socket );
    }
#ifdef WIN32
    WSACleanup();
#endif
}

void PlexNetClient::InitSocketLibrary()
{
#ifdef WIN32
    WSADATA   wsaData;
    WORD wVersionRequested = MAKEWORD( 2, 2 );
    if ( WSAStartup( wVersionRequested, &wsaData ) != 0 ) {
        throw runtime_error( "Unable to init Windows sockets" );
    }
#else
    // do nothing in non-Windows OS
#endif
}

void PlexNetClient::CreateSocket()
{
    m_Socket = socket( AF_INET, ( m_Protocol == ProtocolTCPIP ) ? SOCK_STREAM : SOCK_DGRAM, 0 );
    if ( m_Socket == INVALID_SOCKET ) {
        throw runtime_error( "Unable to create socket" );
    }
}

void PlexNetClient::ConnectSocket()
{
    // get IP address of the PlexNetLocal server
    memset( &m_SocketAddressIn, 0, sizeof( SOCKADDR_IN ) );
    hostent* he = gethostbyname( m_HostAddress.c_str() );
    memcpy( &m_SocketAddressIn.sin_addr, he->h_addr_list[0], he->h_length );

    m_SocketAddressIn.sin_family = AF_INET;

    // port is configurable. look up port number
    // at the top of PlexNetLocal dialog
    m_SocketAddressIn.sin_port = htons( m_HostPort );

    // address of the PlexNetLocal server
    // needs to be in SOCKADDR structure instead of SOCKADDR_IN
    // we simply copy contents of socketAddress into this struct
    memset( &m_SocketAddress, 0, sizeof( m_SocketAddress ) );
    memcpy( &m_SocketAddress, &m_SocketAddressIn, sizeof( m_SocketAddress ) );

    if ( m_Protocol == ProtocolTCPIP ) {
        if ( connect( m_Socket, &m_SocketAddress, sizeof( SOCKADDR ) ) != 0 ) {
            throw runtime_error( "Unable to connect" );
        }
    }
}

void PlexNetClient::SendCommandConnectClient()
{
    // send command to connect the client and set the data transfer mode
    memset( &m_SendBuffer[0], 0, PACKETSIZE );
    m_SendBufferIntPointer[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT;

    // this command specifies general data transfer options: what data types we want to transfer
    // we will need to send additional commands later to specify details of data transfer

    // here we specify that we want everything.
    // we will send other commands later where we will specify what data we want from what channels
    m_SendBufferIntPointer[1] = 1; // want timestamps
    m_SendBufferIntPointer[2] = 1; // want spike waveforms
    m_SendBufferIntPointer[3] = 1; // want analog data
    m_SendBufferIntPointer[4] = 1;  // spike channel from
    m_SendBufferIntPointer[5] = 128; // spike channel to

    SendPacket();
}

void PlexNetClient::SendPacket()
{
    int bytesSentInSingleSend = 0;
    int totalBytesSent = 0;
    int attempt = 0;
    // send may transmit fewer bytes than PACKETSIZE
    // try several sends
    while ( totalBytesSent < PACKETSIZE && attempt < 1000 ) {
        attempt++;
        if ( m_Protocol == ProtocolTCPIP ) {
            bytesSentInSingleSend = send( m_Socket, &m_SendBuffer[0] + totalBytesSent, PACKETSIZE - totalBytesSent, 0 );
            // if ( bytesSentInSingleSend == SOCKET_ERROR ) {
            if ( bytesSentInSingleSend == -1 ) {
                throw runtime_error( "Unable to send: send error" );
            }
        } else { // UDP
            bytesSentInSingleSend = sendto( m_Socket, &m_SendBuffer[0] + totalBytesSent, PACKETSIZE - totalBytesSent, 0, ( SOCKADDR* )&m_SocketAddress, sizeof( m_SocketAddress ) );
            // if ( bytesSentInSingleSend == SOCKET_ERROR ) {
            if ( bytesSentInSingleSend == -1 ) {
                throw runtime_error( "Unable to send: send error" );
            }
        }
        totalBytesSent += bytesSentInSingleSend;
    }
    if ( totalBytesSent != PACKETSIZE ) {
        throw runtime_error( "Unable to send: Cannot send all the bytes of the packet" );
    }
}

void PlexNetClient::ReceivePacket()
{
    SOCKADDR_IN SenderAddr; // for UDP recvfrom only
    socklen_t SenderAddrSize = sizeof( SenderAddr );

    int bytesReceivedInSingleRecv = 0;
    int totalBytesReceived = 0;
    int attempt = 0;
    while ( totalBytesReceived < PACKETSIZE && attempt < 1000 ) {
        attempt++;
        if ( m_Protocol == ProtocolTCPIP ) {
            bytesReceivedInSingleRecv = recv( m_Socket, &m_ReceiveBuffer[0] + totalBytesReceived, PACKETSIZE - totalBytesReceived, 0 );
            // if ( bytesReceivedInSingleRecv == SOCKET_ERROR ) {
            if ( bytesReceivedInSingleRecv == -1 ) {
                throw runtime_error( "Unable to receive: receive error" );
            }
        } else { // UDP
            bytesReceivedInSingleRecv = recvfrom( m_Socket, &m_ReceiveBuffer[0] + totalBytesReceived, PACKETSIZE - totalBytesReceived, 0, ( SOCKADDR* )&SenderAddr, &SenderAddrSize );
            // if ( bytesReceivedInSingleRecv == SOCKET_ERROR ) {
            if ( bytesReceivedInSingleRecv == -1 ) {
                throw runtime_error( "Unable to receive: receive error" );
            }
        }
        totalBytesReceived += bytesReceivedInSingleRecv;
    }
    if ( totalBytesReceived != PACKETSIZE ) {
        throw runtime_error( "Unable to receive: Cannot receive all the bytes of the packet" );
    }
}

void PlexNetClient::ProcessMmfSizesMessage()
{
    // m_ReceiveBufferIntPointer[3] contains the number of supported commands
    int numCommands = m_ReceiveBufferIntPointer[3];
    if ( numCommands > 0 && numCommands < 32 ) {
        for ( int i = 0; i < numCommands; i++ ) {
            if ( m_ReceiveBufferIntPointer[4 + i] == PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS ) {
                m_SupportsSelectSpikeChannels = true;
            }
            if ( m_ReceiveBufferIntPointer[4 + i] == PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS ) {
                m_SupportsSelectContChannels = true;
            }
        }
    }
}

void PlexNetClient::GetPlexNetInfo()
{
    ReceivePacket();
    if ( m_ReceiveBufferIntPointer[0] != PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_MMF_SIZES ) {
        throw runtime_error( "Unable to get PlexNet Info: wrong packet type" );
    }
    ProcessMmfSizesMessage();
    if ( m_SupportsSelectSpikeChannels && m_SupportsSelectContChannels ) {
        RequestParameterMMF();
        GetNumbersOfChannels();
    }
}

void PlexNetClient::RequestParameterMMF()
{
    memset( &m_SendBuffer[0], 0, PACKETSIZE );
    m_SendBufferIntPointer[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF;
    SendPacket();
}

void PlexNetClient::GetNumbersOfChannels()
{
    // read parameters until we get PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA
    bool gotServerArea = false;
    while ( !gotServerArea ) {
        ReceivePacket();
        if ( m_ReceiveBufferIntPointer[0] == PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA ) {
            m_NumSpikeChannels = m_ReceiveBufferIntPointer[15];
            m_NumContinuousChannels = m_ReceiveBufferIntPointer[17];
            gotServerArea = true;
        }
    }
}

void PlexNetClient::SelectSpikeChannels( const std::vector<unsigned char>& channelSelection )
{
    if ( channelSelection.size() > 0 ) {
        SendLongBuffer( PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS, &channelSelection[0], channelSelection.size() );
    }
}

void PlexNetClient::SendLongBuffer( int command, const unsigned char* buffer, int length )
{
    // we have PlexNetMultiMessageTransmissionHeader and the rest is bufLength that we can use for data
    int buflength = PACKETSIZE - sizeof( PlexNetMultiMessageTransmissionHeader );
    int numberOfMessages = length / buflength;
    if ( length % buflength ) {
        numberOfMessages++;
    }
    int positionOfFirstByte = 0;
    for ( int i = 0; i < numberOfMessages; i++ ) {
        memset( &m_SendBuffer[0], 0, PACKETSIZE );
        int numBytesInThisMessage = buflength;
        if ( length % buflength && i == numberOfMessages - 1 ) {
            numBytesInThisMessage = length % buflength;
        }
        int headerSize = sizeof( PlexNetMultiMessageTransmissionHeader );
        PlexNetMultiMessageTransmissionHeader* header = ( PlexNetMultiMessageTransmissionHeader* ) &m_SendBuffer[0];
        header->CommandCode = command;
        header->MessageIndex = i;
        header->NumberOfMessages = numberOfMessages;
        header->NumberOfBytesInThisMessage = numBytesInThisMessage;
        header->PositionOfTheFirstByte = positionOfFirstByte;
        memcpy( &m_SendBuffer[0] + headerSize, buffer + positionOfFirstByte, numBytesInThisMessage );
        positionOfFirstByte += numBytesInThisMessage;
        SendPacket();
    }
}

void PlexNetClient::SelectContinuousChannels( const std::vector<unsigned char>& channelSelection )
{
    if ( channelSelection.size() > 0 )  {
        SendLongBuffer( PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS, &channelSelection[0], channelSelection.size() );
    }
}

void PlexNetClient::StartDataPump()
{
    memset( &m_SendBuffer[0], 0, PACKETSIZE );
    m_SendBufferIntPointer[0] = PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP;
    SendPacket();
}

