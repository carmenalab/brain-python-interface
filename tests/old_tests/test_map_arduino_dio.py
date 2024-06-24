import serial
import struct
import time
port = serial.Serial('/dev/arduino_neurosync', baudrate=115200)
from riglib.dio.parse import MSG_TYPE_ROWBYTE, MSG_TYPE_REGISTER

# Registration
system = 'task'

for sys_name_chr in system:
    reg_word = construct_word(2, MSG_TYPE_REGISTER, ord(sys_name_chr))
    send_data_word_to_serial_port(reg_word)

null_term_word = construct_word(2, MSG_TYPE_REGISTER, 0) # data payload is 0 for null terminator
send_data_word_to_serial_port(null_term_word)

rowcount = 0
ix = np.array([2**i for i in range(16)])
for j in ix:

    for i in range(100):
        #Bit 0: 
        word_str = 'd' + struct.pack('<H', j)
        port.write(word_str)

        #current_sys_rowcount = rowcount
        # rowcount+= 1

        # # construct the data packet
        # print word, current_sys_rowcount
        # word = construct_word(2, MSG_TYPE_ROWBYTE, current_sys_rowcount % 256)
        # send_data_word_to_serial_port(word)
        time.sleep(.02)


def send_data_word_to_serial_port(word):
    word_str = 'd' + struct.pack('<H', word)
    port.write(word_str)


def construct_word(aux, msg_type, data, n_bits_data=8, n_bits_msg_type=3):
    word = (aux << (n_bits_data + n_bits_msg_type)) | (msg_type << n_bits_data) | data
    return word