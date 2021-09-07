FLAGS = -Wall -g -std=gnu99 

all: classifier 

classifier : classifier.o knn.o
	gcc ${FLAGS} -o $@ $^ -lm

test_distance : test_distance.o knn.o
	gcc ${FLAGS} -o $@ $^ -lm


%.o : %.c knn.h
	gcc ${FLAGS} -c $<


.PHONY: clean all

clean:	
	rm classifier test_distance *.o
