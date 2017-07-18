#include"BST.h"

void insert(NODE** root, char* str) {
	NODE** curr = root;
	NODE* p = NULL;
	while (*curr){
		p = *curr;
		if (strcmp((*curr)->key,str)>0){
			curr = &(**curr).left;
		}
		else if (strcmp((*curr)->key, str)<0){
			curr = &(**curr).right;
			//curr=&(*curr).right;
		}
		else{
			return;
		}
	}
	*curr = (NODE*)malloc(sizeof(NODE));
	strcpy((*curr)->key,str);
	(*curr)->left = (*curr)->right = NULL;
	(*curr)->parent = p;
}
NODE* Find(NODE* root, char* key) {
	while (root && strcmp(root->key,key)!=0){
		if (strcmp(root->key , key)>0){
			root = root->left;
		}
		else{
			root = root->right;
		}
	}
	return root;
}
NODE* ToLeft(NODE* node) {
	while (node && node->left){
		node = node->left;
	}
	return node;
}
void TransPlant(NODE** root, NODE* del, NODE* succ) {   //지울곳, 대체자
	//del 는 빠질 노드이므로, left,right,parent는 신경쓰지 않음
	if (del->parent == NULL){   //del 가 root 임
		*root = succ;
	}
	else if (del->parent->left == del){   //del가 부모의 왼쪽자식
		del->parent->left = succ;
	}
	else{
		del->parent->right = succ;
	}
	if (succ){   //succ가 NULL 이 아닐경우 부모도 보장을 해야함
		succ->parent = del->parent;
	}
}
void Erase(NODE** root, char* key) {
	NODE* curr = Find(*root, key);
	if (curr == NULL){   //찾는값이 없을경우 종료
		return;
	}
	else if (curr->left == NULL){
		TransPlant(root, curr, curr->right);
	}
	else if (curr->right == NULL){
		TransPlant(root, curr, curr->left);
	}
	else{
		NODE* succ = ToLeft(curr->right);
		if (succ->parent != curr){               //   (c)->(s) 가 아닐때
			TransPlant(root, succ, succ->right);   //succ를 추출한다.
			succ->right = curr->right;            //c의 R자식이 s 가 아니므로, 오른쪽 자식을 연결
			succ->right->parent = succ;
		}
		TransPlant(root, curr, succ);            //curr을 제거.   
		succ->left = curr->left;               //L자식은 반드시 존재하므로, succ에 옮김
		succ->left->parent = succ;
	}
	free(curr);
}
NODE* begin(NODE* root) {
	return ToLeft(root);
}
NODE* end(NODE* root) {
	return NULL;
}
void ToNext(NODE** pit) {
	NODE* it = *pit;
	if (it->right == NULL){
		while (it->parent && it->parent->right == it){
			it = it->parent;
		}
		it = it->parent;
	}
	else{
		it = ToLeft(it->right);
	}
	*pit = it;
}
void Dealloc(NODE* root) {
	NODE* min = root;
	NODE* temp;
	while (root){
		if (root->right){
			min = ToLeft(root);
			min->left = root->right;
		}
		temp = root;
		root = root->left;
		free(temp);
	}
}