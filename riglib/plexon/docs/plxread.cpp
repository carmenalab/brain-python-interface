/////////////////////////////////////////////////////////////////////
// plxread.cpp - sample functionality for reading PLX files.


#include <windows.h>
#include <stdio.h>
#include "..\Plexon.h"

#define MAX_SPIKE_CHANNELS   (128)
#define MAX_EVENT_CHANNELS   (512)
#define MAX_SLOW_CHANNELS    (256)

#define MAX_SAMPLES_PER_WAVEFORM (256)

// Open PLX file
FILE* fp = 0 ;

// PLX File header structure
PL_FileHeader fileHeader;

// PLX Spike Channel headers
PL_ChanHeader spikeChannels[MAX_SPIKE_CHANNELS];

// PLX Event Channel headers
PL_EventHeader eventChannels[MAX_EVENT_CHANNELS];

// PLX Slow A/D Channel headers
PL_SlowChannelHeader slowChannels[MAX_SLOW_CHANNELS];

// position in file where data begins
int data_start = 0 ;

// Dump header information
bool bDumpHeader = false ;
// Dump spike data blocks
bool bDumpSpike = false ;
// Dump event data blocks
bool bDumpEvent = false ;
// Dump slow A/D data blocks
bool bDumpSlow  = false ;

// Maximum number of data blocks to dump
#define MAX_DATA_BLOCKS (500)

// Dump the file and channel headers.
void DumpHeader ()
{
	int iChannel ;
	int iUnit ;
	
	// Dump the file header
	printf("PLX File Version(%d) Date: %d/%d/%d %d:%d:%d: %s\n",
		fileHeader.Version,
		fileHeader.Month,
		fileHeader.Day,
		fileHeader.Year,
		fileHeader.Hour,
		fileHeader.Minute,
		fileHeader.Second,
		fileHeader.Comment) ;

	printf ("  ADFrequency %d\n", fileHeader.ADFrequency) ;
	printf ("  WaveformFreq %d\n", fileHeader.WaveformFreq) ;
	printf ("  NumPointsWave %d\n", fileHeader.NumPointsWave) ;
	printf ("  NumPointsPerThr %d\n", fileHeader.NumPointsPreThr) ;

	printf ("  LastTimestamp %.0lf\n", fileHeader.LastTimestamp) ;

	if (fileHeader.Version >= 103)
	{
		printf ("  Trodalness %d\n", fileHeader.Trodalness) ;
		printf ("  DataTrodalness %d\n", fileHeader.DataTrodalness) ;
		printf ("  BitsPerSpikeSample %d\n", fileHeader.BitsPerSpikeSample) ;
		printf ("  BitsPerSlowSample %d\n", fileHeader.BitsPerSlowSample) ;
		printf ("  NumDSPChannels %d\n", fileHeader.NumDSPChannels) ;
		printf ("  NumEventChannels %d\n", fileHeader.NumEventChannels) ;
		printf ("  NumSlowChannels %d\n", fileHeader.NumSlowChannels) ;

		printf ("  SpikeMaxMagitudeMV %d\n", fileHeader.SpikeMaxMagnitudeMV) ;
		printf ("  SlowMaxMagnitudeMV %d\n", fileHeader.SlowMaxMagnitudeMV) ;
	}

	// Dump the spike channel counts
	printf ("\nSpike Counts\n") ;
	for (iChannel = 0 ; iChannel < 130 ; iChannel++)
	{
		for (iUnit = 0 ; iUnit < 5 ; iUnit++)
		{
			int count = fileHeader.TSCounts[iChannel][iUnit] ;
			if (count > 0)
			{
				int countWF = fileHeader.WFCounts [iChannel][iUnit] ;
				printf("  Channel %d, Unit %d: Time Stamp %d, Waveform %d\n", iChannel, iUnit, count, countWF) ;
			}
		}
	}

	// Dump the event channel counts
	printf ("\nEvent Counts\n") ;
	for (iChannel = 0 ; iChannel < 300 ; iChannel++)
	{
		int count = fileHeader.EVCounts[iChannel] ;
		if (count > 0)
		{
			printf ("  Index %d Event Channel %d: Count %d\n", iChannel, iChannel, count) ;
		}
	}
	for (iChannel = 300 ; iChannel < 512 ; iChannel++)
	{
		int count = fileHeader.EVCounts[iChannel] ;
		if (count > 0)
		{
			printf ("  Index %d Slow Channel %d: Count %d\n", iChannel, iChannel-300+1, count) ;
		}
	}

	// Dump the spike channel headers
	printf ("\nSpike Data Channels\n") ;
	for (iChannel = 0 ; iChannel < fileHeader.NumDSPChannels ; iChannel++)
	{
		PL_ChanHeader * pSpikeChannel = & spikeChannels[iChannel] ;

		printf ("  Channel %d Name(%s) Sig(%s): WFRate %d, SIG %d Ref %d Gain %d Filter %d Threshold %d\n", 
			pSpikeChannel->Channel,
			pSpikeChannel->Name,
			pSpikeChannel->SIGName,
			pSpikeChannel->WFRate, 
			pSpikeChannel->SIG,
			pSpikeChannel->Ref, 
			pSpikeChannel->Gain, 
			pSpikeChannel->Filter, 
			pSpikeChannel->Threshold) ;

		if (pSpikeChannel->Method == 1)
		{
			int nUnits = pSpikeChannel->NUnits ;
			if (nUnits > 0)
			{
				printf ("    Method: %d - Boxes  NUnits %d\n", 
					pSpikeChannel->Method,
					nUnits) ;

				for (iUnit = 1 ; iUnit < (nUnits+1) ; iUnit++)
				{
					printf("    Boxes[%d][2][4]: (%d,%d,%d,%d) and (%d,%d,%d,%d)\n", iUnit, 
						pSpikeChannel->Boxes[iUnit][0][0],
						pSpikeChannel->Boxes[iUnit][0][1],
						pSpikeChannel->Boxes[iUnit][0][2],
						pSpikeChannel->Boxes[iUnit][0][3],
						pSpikeChannel->Boxes[iUnit][1][0],
						pSpikeChannel->Boxes[iUnit][1][1],
						pSpikeChannel->Boxes[iUnit][1][2],
						pSpikeChannel->Boxes[iUnit][1][3]) ;
				}
			}
		}
		else if (pSpikeChannel->Method == 2)
		{
			int nUnits = pSpikeChannel->NUnits ;
			if (nUnits > 0)
			{
				printf ("    Method: %d - Templates  NUnits %d SortBeg %d SortWidth %d\n", 
					pSpikeChannel->Method,
					nUnits,
					pSpikeChannel->SortBeg,
					pSpikeChannel->SortWidth) ;
	
				for (iUnit = 1 ; iUnit < (nUnits+1) ; iUnit++)
				{
					printf("    Template[%d][]", iUnit) ;

					for (int iSample = 0 ; iSample < fileHeader.NumPointsWave ; iSample++)
					{
						printf(" %d", pSpikeChannel->Template[iUnit][iSample]) ;
					}
					printf("\n") ;
				}
				for (iUnit = 1 ; iUnit < (nUnits+1) ; iUnit++)
				{
					printf("    Fit[%d]: %d\n", iUnit, pSpikeChannel->Fit[iUnit]) ;
				}
			}
		}
		else
		{
			printf("    Method: %d - Unknown\n", pSpikeChannel->Method) ;
			printf("    NUnits: %d\n", pSpikeChannel->NUnits) ;
		}
	}

	// Dump the event channel headers
	printf("\nEvent Channel Headers\n") ;
	for (iChannel = 0 ; iChannel < fileHeader.NumEventChannels ; iChannel++)
	{
		PL_EventHeader * pEventChannel = & eventChannels [iChannel] ;

		printf("    Channel %d Name(%s)\n", 
			pEventChannel->Channel,
			pEventChannel->Name) ;
	}

	// Dump the slow A/D channel headers
	printf ("\nSlow A/D Channel Headers\n") ;
	for (iChannel = 0 ; iChannel < fileHeader.NumSlowChannels ; iChannel++)
	{
		PL_SlowChannelHeader * pSlowChannel = & slowChannels [iChannel] ;

		printf ("    Channel %d Name(%s): ADFreq %d Gain %d Enabled %d PreAmpGain %d\n", 
			pSlowChannel->Channel+1,  // always report to the UI 1-based even though internally it is zero-based
			pSlowChannel->Name,
			pSlowChannel->ADFreq,
			pSlowChannel->Gain,
			pSlowChannel->Enabled,
			pSlowChannel->PreAmpGain) ; 
	}	
}

// Dump the data blocks
void DumpData ()
{
	PL_DataBlockHeader dataBlock;
	short buf[MAX_SAMPLES_PER_WAVEFORM];

	// Seek to the beginning of the data blocks in the PLX file
	fseek(fp, data_start, SEEK_SET);

	int i;
	int nbuf ;

	// Rip through the rest of the file
	for (int iBlock = 0 ; ; iBlock++)
	{
		// Read the next data block header.
		if (fread(&dataBlock, sizeof(dataBlock), 1, fp) != 1) break ;

		// Read the waveform samples if present.
		nbuf = 0;
		if(dataBlock.NumberOfWaveforms > 0)
		{
			nbuf = dataBlock.NumberOfWaveforms*dataBlock.NumberOfWordsInWaveform;
			if (fread(buf, nbuf*2, 1, fp) != 1) break ;
		}

		// Convert the timestamp to seconds
		LONGLONG ts = ((static_cast<LONGLONG>(dataBlock.UpperByteOf5ByteTimestamp)<<32) + static_cast<LONGLONG>(dataBlock.TimeStamp)) ;
		double seconds = (double) ts / (double) fileHeader.ADFrequency ;

		// Dump the spike data block if enabled
		if(bDumpSpike && (dataBlock.Type == PL_SingleWFType))
		{ 
			printf("Spike %d ticks %I64d seconds %.6lf:", 
						dataBlock.Channel,
						ts,
						seconds);
			for(i=0; i<dataBlock.NumberOfWordsInWaveform; i++)
			{
				printf(" %d", buf[i]);
			}
			printf("\n");
		}
		// Dump the event data block if enabled
		else if(bDumpEvent && (dataBlock.Type == PL_ExtEventType))
		{ 
			printf("Event %d ticks %I64d seconds %.6f", 
				dataBlock.Channel,
				ts,
				seconds);
			printf("\n");
		}
		// Dump the slow data block if enabled
		else if(bDumpSlow && (dataBlock.Type == PL_ADDataType))
		{ 
			printf("Slow %d ticks: %I64d seconds %.6f:",
				dataBlock.Channel+1,  // report to the UI 1-based even though internally it is 0-based.
				ts,
				seconds);

			for(i=0; i<dataBlock.NumberOfWordsInWaveform; i++)
			{
				printf(" %d", buf[i]);
			}
			printf("\n");
		}

		// Quit reading data blocks when maximum hit.  Note real applications
		// typicall read all of the data blocks.
		if ((iBlock+1) > MAX_DATA_BLOCKS)
		{
			printf ("Only the first %d data blocks were extracted.", MAX_DATA_BLOCKS) ;
			break ;
		}
	}
	
}

// Print out help to the console if the user does not specify a PLX file to read
void Usage ()
{
	printf ("\n\nplxread Version 2.0 Usage:\n\n") ;
	printf ("> plxread <filename> [-all] [-header] [-spikes] [-events] [-slow]\n") ;
	printf ("  where:\n") ;
	printf ("    -header - dumps the entire header\n") ;
	printf ("    -spike  - extracts and dumps the PL_SingleWFType (Type 1) data blocks\n") ;
	printf ("    -event  - extracts and dumps the PL_ExtEventType (Type 4) data blocks\n") ;
	printf ("    -slow   - extracts and dumps the PL_ADDataType (Type 5) data blocks\n") ;
	printf ("    -all    - extracts and dumps the header and all data blocks\n") ;
	printf ("\n") ;
	printf ("Example:\n\n") ;
	printf ("> plxread ..\\SampleData\\test1.plx -all\n") ;
	exit (1) ;
}

// Main routine
void main(int argc, char *argv[])
{
	// Print out help to the console if the user does not specify a PLX file to read

	if (argc <= 1)
	{
		Usage() ;
	}

	// Open the specified PLX file.
	fp = fopen(argv[1], "rb");
	if(fp == 0){
		printf("Cannot open PLX file (%s).", argv[1]);
		exit(1);
	}

	// Read the file header
	fread(&fileHeader, sizeof(fileHeader), 1, fp);

	// Read the spike channel headers
	if(fileHeader.NumDSPChannels > 0)
		fread(spikeChannels, fileHeader.NumDSPChannels*sizeof(PL_ChanHeader), 1, fp);

	// Read the event channel headers
	if(fileHeader.NumEventChannels> 0)
		fread(eventChannels, fileHeader.NumEventChannels*sizeof(PL_EventHeader), 1, fp);

	// Read the slow A/D channel headers
	if(fileHeader.NumSlowChannels)
		fread(slowChannels, fileHeader.NumSlowChannels*sizeof(PL_SlowChannelHeader), 1, fp);

	// save the position in the PLX file where data block begin
	data_start = sizeof(fileHeader) + fileHeader.NumDSPChannels*sizeof(PL_ChanHeader)
						+ fileHeader.NumEventChannels*sizeof(PL_EventHeader)
						+ fileHeader.NumSlowChannels*sizeof(PL_SlowChannelHeader);


	// Process the program arugments.
	if (argc == 2)
	{
			bDumpHeader = true ;
			bDumpSpike = true ;
			bDumpEvent = true ;
			bDumpSlow = true ;
	}
	else
	{
		for (int iArg = 2 ; iArg < argc ; iArg++)
		{
			if (strcmp(argv[iArg], "-all") == 0)
			{
				bDumpHeader = true ;
				bDumpSpike = true ;
				bDumpEvent = true ;
				bDumpSlow = true ;

			}
			else if (strcmp(argv[iArg], "-header") == 0)
			{
				bDumpHeader = true ;
			}
			else if (strcmp(argv[iArg], "-spike") == 0)
			{
				bDumpSpike = true ;
			}
			else if (strcmp(argv[iArg], "-event") == 0)
			{
				bDumpEvent = true ;
			}
			else if (strcmp(argv[iArg], "-slow") == 0)
			{
				bDumpSlow = true ;
			}
			else
			{
				Usage() ;
			}
		}
	}

	// Dump the header if requested
	if (bDumpHeader)
	{
		DumpHeader () ;
	}

	// Dump the data if requested
	if (bDumpSpike || bDumpEvent || bDumpSlow)
	{
		DumpData () ;
	}

	fclose(fp);
}

