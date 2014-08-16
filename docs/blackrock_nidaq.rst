.. _blackrock_nidaq:

Blackrock Neural Signal Processor (NSP) digital input
=====================================================

Right now, this is a just a informal set of notes.

We use the Blackrock NSP's digital input port for 2 purposes:
1) Remote recording control (starting/stopping/resuming recording remotely)
2) Writing information into the digital input port during the experiment so that timestamps corresponding to the arrival of various data (ArmAssist feedback data, ReHand feedback data, EMG data) are saved into the Blackrock .nev file for later comparison with spike timestamps.

The NSP's digital input port has 16 data pins (D0 through D15) and one digital strobe (DS) pin. See the NeuroPort manual for more details.

In Central, the NSP should be configured to read the values of the 16 data pins whenever the strobe pin goes from low to high (i.e., has a rising edge). You can do this by clicking "Hardware Configuration" --> Digital In --> "digin" and then selecting "16-bit on word strobe" in the drop-down menu. To avoid having to set this every time the NSP is turned on, you can set it once, then click "File --> Save System Settings" and save a .ccf configuration file. Then, simply load the configuration file every time you open Central (the configuration file can be used to save many other types of settings too).

For remote recording control, use the following settings (click "File Storage --> Remote Recording Control"):
Start:       Digital Bit Input, D15, high
New Session: (leave disabled)
Stop:        Digital Bit Input, D15, low
Resume:      Digital Bit Input, D15, high

Here is how things are connected:
Linux PC's PCI slot --> NI PCI-6503 card --> NI ribbon cable --> NI connector block --> individual wires --> DB37 solder cup adapter --> DB37 cable --> NSP digital input

We use the following National Instruments (NI) digital input/output card: NI PCI-6503. We use the linux "comedi" library to interact with it. The table below shows exactly how the different pins on the NSP's digital input map to the pin #s from comedi's pt of view.
                
NIDAQ hardware interface for Blackrock system
=============================================

=====================    ========    ==========    ===========================    ============    ================================
NSP digital input pin    DB37 pin    wire color    NI 6503 connector block pin    NI 6503 card    pin # (from comedi's pt of view)
=====================    ========    ==========    ===========================    ============    ================================
DS (digital strobe)      1           white         15                             PC0             17
D0                       2           grey          47                             PA0             1
D1                       3           purple        45                             PA1             2
D2                       4           blue          43                             PA2             3
D3                       5           green         41                             PA3             4
D4                       6           yellow        39                             PA4             5
D5                       7           orange        37                             PA5             6
D6                       8           red           35                             PA6             7
D7                       9           brown         33                             PA7             8
D8                       10          black         31                             PB0             9
D9                       11          white         29                             PB1             10
D10                      12          grey          27                             PB2             11
D11                      13          purple        25                             PB3             12
D12                      14          blue          23                             PB4             13
D13                      15          green         21                             PB5             14
D14                      16          yellow        19                             PB6             15
D15                      17          orange        17                             PB7             16
digital ground           GND         black         50