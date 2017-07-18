#ifndef BST_H
#define BST_H
#pragma warning (disable:4996)
#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<memory.h>
#include<string.h>

typedef struct NODE NODE;
struct NODE{
	char key[32];
	NODE* left;
	NODE* right;
	NODE* parent;
	int num;
};

void insert(NODE** root, char* str);
NODE* Find(NODE* root, char* key);
NODE* ToLeft(NODE* node);
void TransPlant(NODE** root, NODE* del, NODE* succ);
void Erase(NODE** root, int key);
NODE* begin(NODE* root);
NODE* end(NODE* root);
void ToNext(NODE** pit);
void Dealloc(NODE* root);

/*
int main() {
	NODE* root = NULL;
	int i;
	for (i = 0; i < 10000; i++){
		insert(&root, i);
	}
	printf("\n\n");

	NODE* it;
	for (it = begin(root); it != end(root); ToNext(&it)){
		printf("%d\n", it->key);
	}

	Dealloc(root);


	return 0;
}
*/
#endif