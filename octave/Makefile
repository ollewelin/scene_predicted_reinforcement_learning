CC = g++
CFLAGS = -Wall -Wextra -pedantic -std=c++17

all: converter q_converter

converter: log_convert.o
	$(CC) $(CFLAGS) log_convert.o -o converter

log_convert.o: log_convert.cpp
	$(CC) $(CFLAGS) -c log_convert.cpp

q_converter: q_log_convert.o
	$(CC) $(CFLAGS) q_log_convert.o -o q_converter

q_log_convert.o: q_log_convert.cpp
	$(CC) $(CFLAGS) -c q_log_convert.cpp

clean:
	rm -f *.o converter
