CC=mpicc
CFLAGS=-fopenmp -O2

SRC=src/main.c src/parser.c src/model.c src/utils.c
OUT=log_monitor

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)
