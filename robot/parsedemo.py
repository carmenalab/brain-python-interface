from numpy import *
import textwrap;
import sys;
import re;

# we use termios to pretty up the continuation prompting, only available under Unix
has_termios = False;
try:
    import termios;
    has_termios = True;
except:
    pass;
    
def parsedemo(s):
    """
    Helper to write demo files, based on minimum change from their Matlab format.
    The string contains
      - help text which is left justified and displayed directly to the screen
      - indented text is interpretted as a command, executed, and the output sent to
        screen
        
    The argument is Matlab code as per the demo files rtXXdemo.
    
    @note: Requires some minor changes to the demo file.
    @type s: string
    @param s: Demo string.
    """
    rstrip = re.compile(r'''\s*%.*$''')
    lines = s.split('\n');
    
    name = __file__;
    if has_termios:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] = new[3] & ~termios.ECHO          # lflags
    try:
        if has_termios:
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
            
        text = '';
        for line in lines:
            if len(line) == 0:
                print;
            elif line[0] == '#':
                ## help text found, add it to the string
                text += line[1:].lstrip() + '\n';
                
                # a blank line means paragraph break
                if len(line) == 1 and text:
                    text = '';
            else:
                ## command encountered
                
                # flush the remaining help text
                if text:
                    text = '';
                
                cmd = line.strip();
                
                # special case, pause prompts the user to continue
                if cmd.startswith('pause'):
                    # prompt for continuation
                    sys.stderr.write('more? ')
                    raw_input();
                    
                    # remove the prompt from the screen
                    sys.stderr.write('\r        \r');
                    continue;
                
                # if it involves an assignment then we use exec else use eval.
                # we mimic the matlab behaviour in which a trailing semicolon inhibits
                # display of the result
                if cmd.startswith('from'):
                    exec(cmd)
                elif '=' in cmd:
                    e = cmd.split('=');
                    exec(rstrip.sub('', cmd))

                else:
                    result = eval(cmd.rstrip(';'));


    finally:
        if has_termios:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
