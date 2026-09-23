CC = gcc 
#orig CC = clang++
#-std=c11
CFLAGS = -Wall -mavx2
# DEBUG mode
#CFLAGS += -g
# RELEASE mode
CFLAGS += -O2 -DNDEBUG
CFLAGS += -D__GATHER__=1 # __GATHER__=1 use gather instruction, example _mm256_i32gather_epi32, =0, use set instructions _mm256_set_epi64x
LDFLAGS =

#APP_NAME = TransposeBitMatrix_test
# Проект 1 benchmark Usuba реализаций
PROJ1_SRCS = Transpose_Usuba_main.c
PROJ1_OBJS = $(PROJ1_SRCS:.c=.o)
PROJ1_BIN = TransposeBitMatrix_Usuba_test

# Проект 2 benchmark моих реализаций 
#  TransposeBitMatrix_macros.h TransposeBitMatrix_Alg1.h Settings.h Transpose_8x8_movemask.h
PROJ2_SRCS = TransposeBitMatrix_main.c TransposeBitMatrix_Alg1.c Transpose_8x8_movemask.c
PROJ2_OBJS = $(PROJ2_SRCS:.c=.o)
PROJ2_BIN = TransposeBitMatrix_test

all: $(PROJ1_BIN) $(PROJ2_BIN)

$(PROJ1_BIN): $(PROJ1_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(PROJ2_BIN): $(PROJ2_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(PROJ1_OBJS) $(PROJ2_OBJS) $(PROJ1_BIN) $(PROJ2_BIN)

