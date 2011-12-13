import ftdi

class Button(object):
    def __init__(self):
        self.port = None
        port = ftdi.ftdi_new()
        usb_open = ftdi.ftdi_usb_open_string(port, "s:0x403:0x6001:2eb80091")
        assert usb_open == 0, ftdi.ftdi_get_error_string(port)
        
        ftdi.ftdi_set_bitmode(port, 0xFF, ftdi.BITMODE_BITBANG)
        self.port = port
        self.last = self._check()
    
    def _check(self):
        test = ' '
        ftdi.ftdi_read_pins(self.port, test)
        return ord(test)
    
    def pressed(self):
        status = False
        cur = self._check()
        if cur in [1, 2] and self.last == 0:
            status = cur
            
        self.last = cur
        return status
    
    def __del__(self):
        if self.port is not None:
            ftdi.ftdi_disable_bitbang(self.port)
            ftdi.ftdi_usb_close(self.port)
            ftdi.ftdi_deinit(self.port)

