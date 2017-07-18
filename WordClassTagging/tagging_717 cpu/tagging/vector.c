#include"vector.h"

void vec_create(VECTOR** head){
	*head = (VECTOR*)malloc(1 * sizeof(VECTOR));
	(*head)->cap = 0; (*head)->size = 0; (*head)->element = NULL;
}

void vec_pushback(VECTOR* head, data n_data){
	
	if (head->cap != head->size){
		memcpy(&(head->element[head->size]), &n_data, sizeof(data));
		head->size++;
	}
	else{
		data* n_elem;
		head->cap = head->cap != 0 ? 2 * head->cap : 2;
		n_elem = (data*)malloc(head->cap*sizeof(data));
		memcpy(n_elem, head->element, head->size*sizeof(data));
		memcpy(&(n_elem[head->size]), &n_data, sizeof(data));
		free(head->element);
		head->element = n_elem;
		head->size++;
	}


}