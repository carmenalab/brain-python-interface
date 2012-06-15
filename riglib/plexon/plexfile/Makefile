CC=gcc
CFLAGS=-O3 -march=native -Wall -Wextra -fPIC -c
LDFLAGS=-lm
SOURCES=inspect.c plexfile.c dataframe.c plexread.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=test

all: $(SOURCES) $(EXECUTABLE) lib

lib: $(OBJECTS)
	$(CC) $(OBJECTS) -shared -o plexfile.so
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE)
