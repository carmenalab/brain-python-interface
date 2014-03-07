//
// RawSocket.cpp
//
//    Console-mode app demonstrating how to receive PlexNet packets using either
//    TCP/IP or UDP.  Note that a blocking socket is used for simplicity,
//    but a non-blocking socket could be used as well. See PlexNetClient.cpp 
//    and .h for details of the class PlexNetClient. For questions or support, 
//    please contact support@plexon.com.
//
// Author: Alex Kirillov
// (c) 1998-2014 Plexon Inc (www.plexon.com)
//
#include <stdio.h>
#include <vector>
#include <iostream>
#include "PlexNetClient.h"

#include <fstream>

void ProcessPacket( const std::vector<char>& receiveBuffer, PL_WaveLong& wave);

using namespace std;

int main( int argc, char* argv[] )
{
    // IP address of the remote machine to which the OPX/MAP system is connected;
    // substitute the actual IP address of your machine running PlexNetLocal
    // look up IP address at the top of PlexNetLocal dialog
    string hostAddress = "10.0.0.13";

    // Port used by PlexNet on the machine to which the OPX/MAP system is connected.
    // It can be set in the PlexNetLocal options dialog.
    // look up port number at the top of PlexNetLocal dialog
    int port = 6000;

    // also, specify which protocol to use (PlexNetClient::ProtocolTCPIP or PlexNetClient::ProtocolUDP)
    //PlexNetClient::Protocol theProtocol = PlexNetClient::ProtocolUDP;
    PlexNetClient::Protocol theProtocol = PlexNetClient::ProtocolTCPIP;

    try {
        PlexNetClient client( hostAddress.c_str(), port, theProtocol );
        client.InitSocketLibrary();
        client.CreateSocket();

        cout << "Connecting to PlexNet..." << endl;
        client.ConnectSocket();
        cout << "Connected to PlexNet at address " << hostAddress.c_str() << ", port " <<  port << endl;

        client.SendCommandConnectClient();
        client.GetPlexNetInfo();

        cout << "PlexNet supports spike channel selection: " << client.SupportsSelectSpikeChannels() << endl;
        cout << "PlexNet supports continuous channel selection: " << client.SupportsSelectSpikeChannels() << endl;
        cout << "PlexNet has " << client.NumSpikeChannels() << " spike channels"  << endl;
        cout << "PlexNet has " << client.NumContinuousChannels() << " continuous channels"  << endl;

        // specify what data we want for spike channels
        if ( client.NumSpikeChannels() > 0 ) {
            vector<unsigned char> spikeChannelsSelection( client.NumSpikeChannels() );
            memset( &spikeChannelsSelection[0], 0, spikeChannelsSelection.size() );
            // we have 1 byte for each channel in spikeChannelsSelection vector
            // in each byte we can specify any combination of the following flags:
            // #define SPIKE_CHAN_SORTED_TIMESTAMPS (0x01)
            // #define SPIKE_CHAN_SORTED_WAVEFORMS (0x02)
            // #define SPIKE_CHAN_UNSORTED_TIMESTAMPS (0x04)
            // #define SPIKE_CHAN_UNSORTED_WAVEFORMS (0x08)

            // the following for loop instructs PlexNetLocal to send everything from every spike channel
            //for ( int i = 0; i < spikeChannelsSelection.size(); i++ ) {
            //    spikeChannelsSelection[i] = SPIKE_CHAN_SORTED_TIMESTAMPS | SPIKE_CHAN_SORTED_WAVEFORMS | SPIKE_CHAN_UNSORTED_TIMESTAMPS | SPIKE_CHAN_UNSORTED_WAVEFORMS;
            //}

            // here is an example on how to specify data types in mode detail:
            // example 1: send only sorted timestamps for the first spike channel:
            //int spikeChannel = 1;
            //spikeChannelsSelection[spikeChannel - 1] = SPIKE_CHAN_SORTED_TIMESTAMPS;
            //// example 2: send sorted timestamps and waveforms for the second spike channel:
            //spikeChannel = 2;
            //spikeChannelsSelection[spikeChannel - 1] = SPIKE_CHAN_SORTED_TIMESTAMPS | SPIKE_CHAN_SORTED_WAVEFORMS;
            //// example 3: send sorted and unsorted timestamps and waveforms for spike channel 5:
            //spikeChannel = 5;
            //spikeChannelsSelection[spikeChannel - 1] = SPIKE_CHAN_SORTED_TIMESTAMPS | SPIKE_CHAN_SORTED_WAVEFORMS | SPIKE_CHAN_UNSORTED_TIMESTAMPS | SPIKE_CHAN_UNSORTED_WAVEFORMS;

            // select all spike channels
            for ( int i = 0; i < spikeChannelsSelection.size(); i++ ) {
                spikeChannelsSelection[i] = SPIKE_CHAN_SORTED_TIMESTAMPS | SPIKE_CHAN_SORTED_WAVEFORMS | SPIKE_CHAN_UNSORTED_TIMESTAMPS | SPIKE_CHAN_UNSORTED_WAVEFORMS;
            }
            client.SelectSpikeChannels( spikeChannelsSelection );
        }

        if ( client.NumContinuousChannels() > 0 ) {
            vector<unsigned char> contChannelsSelection( client.NumContinuousChannels() );
            memset( &contChannelsSelection[0], 0, contChannelsSelection.size() );
            // specify what continuous channels we want to receive
            // to get the list of continuous channels,
            // please run PlexNetRemote on the same computer as PlexNexLocal, connect,
            // press Data Transfer Options and tab to Continuous Channels tab
            // channels are listed with 1-based channel numbers

            // the following for loop instructs PlexNetLocal to send all continuous channels
            //for ( int i = 0; i < contChannelsSelection.size(); i++ ) {
            //    contChannelsSelection[i] = 1;
            //}

            // example 1: get only channels 1, 2, and 17
            //int contChannel = 1;
            //contChannelsSelection[contChannel - 1 ] = 1;
            //contChannel = 2;
            //contChannelsSelection[contChannel - 1 ] = 1;
            //contChannel = 17;
            //contChannelsSelection[contChannel - 1 ] = 1;

            // select all continuous channels
            // for ( int i = 0; i < contChannelsSelection.size(); i++ ) {
            //     contChannelsSelection[i] = 1;
            // }
            int chan = 512+8;

            contChannelsSelection[chan-1] = 1;
            client.SelectContinuousChannels( contChannelsSelection );

            client.StartDataPump();

            // short data[10000] = {}
            ofstream foo("plexnet_cpp_data_3.dat",ios::out | ios::binary);
            int n_pts = 0;

            // receive and process data
            int numberOfPacketsToProcess = 2000;
            for ( int i = 0; i < numberOfPacketsToProcess; i++ ) {
                client.ReceivePacket();
                // ProcessPacket( client.GetReceiveBuffer() );
                PL_WaveLong wave;
                ProcessPacket( client.GetReceiveBuffer() , wave);
                if (wave.Channel == chan-1) {
                    foo.write((char *)&wave.WaveForm, wave.NumberOfDataWords*sizeof(short));
                }
            }
            foo.close();
        }
        return 1;
    } catch ( exception ex ) {
        cout << "Exception: " << ex.what() << endl;
    } catch ( ... ) {
        cout << "Unknown exception" << endl;
    }
    return 0;
}

void ProcessPacket( const std::vector<char>& receiveBuffer, PL_WaveLong& wave)
{
    if ( receiveBuffer.size() == 0 ) {
        return;
    }

    // decode the buffer
    const int* ibuf = ( int* )&receiveBuffer[0];
    int NumServerDropped = ibuf[2];
    int NumMMFDropped = ibuf[3];

    PL_DataBlockHeader db;
    // PL_WaveLong wave;
    int sizeOfDataBlockHeader = sizeof( db );
    int sizeOfWaveLong = sizeof( PL_WaveLong );

    short buf[128];
    int nwaves = 0;
    int nts = 0;
    int found_eop = 0;

    int length = PACKETSIZE - 16; // type and msg count
    int pos = 16;

    while ( pos + sizeOfDataBlockHeader <= PACKETSIZE ) {
        // extract one PL_DataBlockHeader
        memcpy( &db, &receiveBuffer[0] + pos, sizeOfDataBlockHeader );

        if ( db.Type == 0 ) {// empty block
            break;
        }

        if ( db.Type == -1 ) { // end of packet, we're done
            found_eop = 1;
            break;
        }

        pos += sizeOfDataBlockHeader;
        int nbuf = 0;

        if ( db.NumberOfWaveforms > 0 ) {
            // get the waveform values
            nbuf = db.NumberOfWaveforms * db.NumberOfWordsInWaveform;
            memcpy( buf, &receiveBuffer[0] + pos, nbuf * 2 );
            pos += nbuf * 2;
        }

        // fill in a PL_WaveLong from the packet
        memset( &wave, 0, sizeOfWaveLong );
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
            if ( nbuf > 0 ) { // it has waveform values
                nwaves++;
            } else { // no waveform values, timestamp only
                nts++;
            }
        }
        if ( wave.Type == PL_ExtEventType ) {
            nts++;
        }
        // cout << "type=" << ( int )wave.Type << " channel=" << ( int )wave.Channel << " unit=" << ( int )wave.Unit << " t=" << wave.TimeStamp << endl;
    }
}
