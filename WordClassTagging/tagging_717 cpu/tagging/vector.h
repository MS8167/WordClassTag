#ifndef VECTOR_H_0133_C713
#define VECTOR_H_0133_C713
#include<stdio.h>
#include<stdlib.h>
#include<memory.h>

typedef struct data{
	int word_num;
	int goal;
}data;
typedef struct VECTOR{
	int cap;
	int size;
	data* element;
}VECTOR;


void vec_create(VECTOR** vec);
void vec_pushback(VECTOR* vec, data n_data);


#endif