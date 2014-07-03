NIDAQ hardware interface
========================

============        =======        ====================     =========   =========   ==================      =======     ============
DIO function        DIO pin        DIO color                NIDAQ pin   NIDAQ pin   DIO color               DIO pin     DIO function
============        =======        ====================     =========   =========   ==================      =======     ============
--                  --             empty                    1           2           --                      --          --    
--                  --             empty                    3           4           --                      --          --    
unused              17             white/violet             5           6           --                      --          --    
RSTART              24             white/black/green        7           8           --                      --          --    
--                  --             empty                    9           10          --                      --          --    
--                  --             empty                    11          12          --                      --          --    
Strobe              22             white/black/orange       13          14          --                      --          --    
--                  --             empty                    15          16          --                      --          --    
Data Bit 15         16             white/blue               17          18          --                      --          --    
Data Bit 14         15             white/green              19          20          --                      --          --    
Data Bit 13         14             white/yellow             21          22          --                      --          --    
Data Bit 12         13             white/orange             23          24          --                      --          --    
Data Bit 11         12             white/red                25          26          --                      --          --    
Data Bit 10         11             white/brown              27          28          --                      --          --    
Data Bit 9          10             black                    29          30          white/black/violet      26          unused
Data Bit 8          9              white                    31          32          white/gray              18          unused
Data Bit 7          8              gray                     33          34          white/black             19          GND
Data Bit 6          7              violet                   35          36          white/black/blue        25          GND
Data Bit 5          6              blue                     37          38          white/black/yellow      23          GND
Data Bit 4          5              green                    39          40          white/black/red         21          GND
Data Bit 3          4              yellow                   41          42          --                      --          --    
Data Bit 2          3              orange                   43          44          --                      --          --    
Data Bit 1          2              red                      45          46          --                      --          --    
Data Bit 0          1              brown                    47          48          --                      --          --    
--                  --             +5V from NIDAQ board     49          50          --                      --          --    
============        =======        ====================     =========   =========   ==================      =======     ============



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