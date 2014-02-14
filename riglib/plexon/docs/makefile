CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=RawSocket_new.cpp PlexNetClient.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=example

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@