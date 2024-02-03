CC = g++
CFLAGS = -Wall -Wextra -pedantic -std=c++17

all: converter

converter: log_convert.o
	$(CC) $(CFLAGS) log_convert.o -o converter

log_convert.o: log_convert.cpp
	$(CC) $(CFLAGS) -c log_convert.cpp

clean:
	rm -f *.o converter
