'''
Interact with a peripheral 'button' device hooked up to an FTDI chip
'''

import threading
import queue
import ftdi
import time

class Button(threading.Thread):
    ''' Docstring '''
    def __init__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(Button, self).__init__()
        self.port = None
        port = ftdi.ftdi_new()
        usb_open = ftdi.ftdi_usb_open_string(port, "s:0x403:0x6001:2eb80091")
        assert usb_open == 0, ftdi.ftdi_get_error_string(port)
        
        ftdi.ftdi_set_bitmode(port, 0xFF, ftdi.BITMODE_BITBANG)
        self.port = port
        self.queue = queue.Queue()
        self.daemon = True
        self.start()
       
    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        last = None
        while True:
            k = self._check()
            if last == 0 and k != 0:
                self.queue.put(k)
            last = k
            time.sleep(0.01)
    
    def _check(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        test = ' '
        ftdi.ftdi_read_pins(self.port, test)
        return ord(test)
    
    def pressed(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        try:
            return self.queue.get_nowait()
        except:
            return None
    
    def __del__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if self.port is not None:
            ftdi.ftdi_disable_bitbang(self.port)
            ftdi.ftdi_usb_close(self.port)
            ftdi.ftdi_deinit(self.port)

if __name__ == "__main__":
    import time
    btn = Button()
    while True:
        k = btn.pressed()
        if k is not None:
            print(k)
