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
void TransPlant(NODE** root, NODE* del, NODE* succ) {   //�����, ��ü��
	//del �� ���� ����̹Ƿ�, left,right,parent�� �Ű澲�� ����
	if (del->parent == NULL){   //del �� root ��
		*root = succ;
	}
	else if (del->parent->left == del){   //del�� �θ��� �����ڽ�
		del->parent->left = succ;
	}
	else{
		del->parent->right = succ;
	}
	if (succ){   //succ�� NULL �� �ƴҰ�� �θ� ������ �ؾ���
		succ->parent = del->parent;
	}
}
void Erase(NODE** root, char* key) {
	NODE* curr = Find(*root, key);
	if (curr == NULL){   //ã�°��� ������� ����
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
		if (succ->parent != curr){               //   (c)->(s) �� �ƴҶ�
			TransPlant(root, succ, succ->right);   //succ�� �����Ѵ�.
			succ->right = curr->right;            //c�� R�ڽ��� s �� �ƴϹǷ�, ������ �ڽ��� ����
			succ->right->parent = succ;
		}
		TransPlant(root, curr, succ);            //curr�� ����.   
		succ->left = curr->left;               //L�ڽ��� �ݵ�� �����ϹǷ�, succ�� �ű�
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