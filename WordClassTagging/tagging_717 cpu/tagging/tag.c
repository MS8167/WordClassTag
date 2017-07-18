#include"BST.h"
#include"vector.h"

#include <math.h>      // for sigmoid
#include <time.h>      // for random generate
#include <stdlib.h>



#define SIG(X) (1/(1+expf((-1*(X)))))

#define STR 4096
#define WRD 64

#define word_vector 50

#define N_LAYER 3         //number of layer

#define N_X_TLU 150      //number of input layer's TLU   3*word_vector
#define N_H_TLU 1024      //number of hidden layer's TLU
#define N_F_TLU 45      //number of final layer's TLU
#define N_MAX_TLU 2000     //max number of TLU per layer

const float LEARNING_RATE = (float)0.02;

// array 분리
float in_layer_0[N_X_TLU];
float in_layer_1[N_H_TLU];
float in_layer_2[N_F_TLU];

float bias_layer_0[N_X_TLU];
float bias_layer_1[N_H_TLU];
float bias_layer_2[N_F_TLU];

float f_layer_0[N_X_TLU];
float f_layer_1[N_H_TLU];
float f_layer_2[N_F_TLU];

float delta_layer_0[N_X_TLU];
float delta_layer_1[N_H_TLU];
float delta_layer_2[N_F_TLU];

float weight_layer0_to_layer1[N_X_TLU][N_H_TLU];
float weight_layer1_to_layer2[N_H_TLU][N_F_TLU];

float soft_max[N_F_TLU];
int n;

typedef struct dictionary{
	float word_vec[50];
	int goal;
}dictionary;
typedef struct vector_window{
	float* prev;
	float* curr;
	float* next;
}vector_window;

int word_cnt;

void file_read(NODE** root);
void numbering(NODE** root, dictionary** dict);
void file_write(NODE* root, VECTOR** head, dictionary* dict);
void dictionary_genaerate(dictionary* dict);
void bias_generate();
void weight_generate();
void epoch(VECTOR* head, dictionary* dict);
int forward_computing(vector_window* window, int goal);
void backward_computing(vector_window* window, int goal, int idx);
void update(vector_window* window);
double test(VECTOR* head, dictionary* dict);


int main(){
	FILE* fpfp;
	NODE* root = NULL;
	VECTOR* head = NULL;
	dictionary* dict = NULL;
	int i = 0;
	double start = 0;
	file_read(&root);
	numbering(&root, &dict);
	file_write(root, &head, dict);


	weight_generate();
	bias_generate();
	

	while (1){
		fpfp = fopen("DEBUG.txt", "a");
		start = clock();
		epoch(head, dict);
		start = clock() - start;
		fprintf(fpfp,"CPU epoch time : %lf\n", start);
		fprintf(fpfp,"%d precise: %lf\n", ++i, test(head, dict));
		printf(".\n");
		fclose(fpfp);
	}

	return 0;
}

void dictionary_genaerate(dictionary* dict){         //   initialize dictionary

	int sign = 1;
	float rand_val = 0;
	//FILE* fp = fopen("word_dictionary_init.txt", "w");
	srand((unsigned)time(NULL));

	for (int i = 1; i <= word_cnt; i++){
		for (int j = 0; j < word_vector; j++){
			rand_val = (float)rand() / RAND_MAX;
			sign = rand()>(RAND_MAX / 2) ? -1 : 1;
			(dict + i)->word_vec[j] = rand_val*sign;
			//   fprintf(fp, "%.8f", (dict + i)->word_vec[j]);
		}
		//fputs("\n", fp);
	}
	//fclose(fp);
}
void bias_generate(){
	int sign = 1;
	float rand_val = 0;
	srand((unsigned)time(NULL));

	/*for (int i = 0; i < N_LAYER - 1; i++){
	for (int j = 0; j < TLU[i]; j++){
	rand_val = (float)rand() / RAND_MAX;
	sign = rand()>RAND_MAX / 2 ? -1 : 1;
	bias[i][j] = sign*rand_val;
	}
	}*/

	for (int i = 0; i < N_X_TLU; i++){
		rand_val = (float)rand() / RAND_MAX;
		sign = rand()>RAND_MAX / 2 ? -1 : 1;
		bias_layer_0[i] = sign*rand_val;
	}
	for (int i = 0; i < N_H_TLU; i++){
		rand_val = (float)rand() / RAND_MAX;
		sign = rand()>RAND_MAX / 2 ? -1 : 1;
		bias_layer_1[i] = sign*rand_val;
	}
	for (int i = 0; i < N_F_TLU; i++){
		rand_val = (float)rand() / RAND_MAX;
		sign = rand()>RAND_MAX / 2 ? -1 : 1;
		bias_layer_2[i] = sign*rand_val;
	}
}
void weight_generate(){                     //   initialize weight (-1,1)
	int sign = 1;
	float rand_val = 0;
	srand((unsigned)time(NULL));

	/*for (int i = 0; i < N_LAYER - 1; i++){
	for (int j = 0; j < TLU[i + 1]; j++){
	for (int k = 0; k < TLU[i]; k++){
	rand_val = (float)rand() / RAND_MAX;
	sign = rand()>RAND_MAX / 2 ? -1 : 1;
	weight[i + 1][j][k] = sign*rand_val;
	}
	}
	}*/
	for (int i = 0; i < N_X_TLU; i++){
		for (int j = 0; j < N_H_TLU; j++){
			rand_val = (float)rand() / RAND_MAX;
			sign = rand()>RAND_MAX / 2 ? -1 : 1;
			weight_layer0_to_layer1[i][j] = sign*rand_val;
		}
	}
	for (int i = 0; i < N_H_TLU; i++){
		for (int j = 0; j < N_F_TLU; j++){
			rand_val = (float)rand() / RAND_MAX;
			sign = rand()>RAND_MAX / 2 ? -1 : 1;
			weight_layer1_to_layer2[i][j] = sign*rand_val;
		}
	}
}
int forward_computing(vector_window* window, int goal){
	float expsum_final_out = 0;
	float error = 0;
	int max_f_idx = 0;      //for modified softmax
	float max_f = 0;
	int i = 0;

	//for (int i = 0; i < 50; i++){
	//   in[0][i] = window->prev[i] + bias[0][i];
	//   f[0][i] = SIG(in[0][i]);
	//}
	//for (int i = 50; i < 100; i++){
	//   in[0][i] = window->curr[i-50] + bias[0][i];
	//   f[0][i] = SIG(in[0][i]);
	//}
	//for (int i = 100; i < N_X_TLU; i++){
	//   in[0][i] = window->next[i-100] + bias[0][i];
	//   f[0][i] = SIG(in[0][i]);
	//}
	for (; i < 50; i++){
		in_layer_0[i] = window->prev[i] + bias_layer_0[i];
		f_layer_0[i] = SIG(in_layer_0[i]);
	}
	for (; i < 100; i++){
		in_layer_0[i] = window->curr[i - 50] + bias_layer_0[i];
		f_layer_0[i] = SIG(in_layer_0[i]);
	}
	for (; i < 150; i++){
		in_layer_0[i] = window->next[i - 100] + bias_layer_0[i];
		f_layer_0[i] = SIG(in_layer_0[i]);
	}

	//0->1 
	for (int i = 0; i < N_H_TLU; i++){
		float weighted_sum = 0.f;
		for (int j = 0; j < N_X_TLU; j++){
			weighted_sum += weight_layer0_to_layer1[j][i] * f_layer_0[j];
		}
		in_layer_1[i] = weighted_sum + bias_layer_1[i];      //in = W * f + b
		f_layer_1[i] = SIG(in_layer_1[i]);               //f =SIG(in)
	}
	//1->2
	for (int i = 0; i < N_F_TLU; i++){
		float weighted_sum = 0.f;
		for (int j = 0; j < N_H_TLU; j++){
			weighted_sum += weight_layer1_to_layer2[j][i] * f_layer_1[j];
		}
		f_layer_2[i] = in_layer_2[i] = weighted_sum + bias_layer_2[i];
		if (f_layer_2[max_f_idx] < f_layer_2[i])max_f_idx = i;
	}
	max_f = f_layer_2[max_f_idx];

	//softmax층 
	for (i = 0; i < N_F_TLU; i++){
		expsum_final_out += expf(f_layer_2[i] - max_f);
	}
	for (i = 0; i < N_F_TLU; i++){
		soft_max[i] = expf(f_layer_2[i] - max_f) / expsum_final_out;
	}
	error = logf(expsum_final_out) - f_layer_2[goal - 1];

	//if (n++ % 1000 == 0)printf("%d\n", n);
	//============================================================before

	//for (int l = 1; l < N_LAYER; l++){
	//   for (int j = 0; j < TLU[l]; j++){
	//      float weighted_sum = 0;
	//      for (int i = 0; i < TLU[l - 1]; i++){
	//         weighted_sum += weight[l][j][i] * f[l - 1][i];
	//      }
	//      //in=b+ w*f
	//      in[l][j] = weighted_sum + bias[l][j];

	//      // f= sig(in)
	//      if (l != N_LAYER - 1){
	//         f[l][j] = SIG(in[l][j]);
	//      }
	//      else{      // final layer
	//         f[l][j] = in[l][j];
	//         if (f[N_LAYER - 1][max_f_idx] < f[N_LAYER - 1][j])max_f_idx = j;         //최대값을 찾는다.
	//      }
	//   }
	//}
	//max_f = f[N_LAYER - 1][max_f_idx];

	//for (int i = 0; i < TLU[N_LAYER - 1]; i++){
	//   expsum_final_out += expf(f[N_LAYER - 1][i] - max_f);
	//}

	//// soft-max 
	//for (int i = 0; i < TLU[N_LAYER - 1]; i++){
	//   soft_max[i] = expf(f[N_LAYER - 1][i] - max_f) / expsum_final_out;
	//}

	//error = logf(expsum_final_out) - f[N_LAYER - 1][goal - 1];
	//return error;

	if (max_f_idx + 1 != goal) return 0;
	else return 1;
}
void backward_computing(vector_window* window, int goal, int idx){
	for (int i = 0; i < N_F_TLU; i++){
		delta_layer_2[i] = soft_max[i];
	}
	delta_layer_2[goal - 1] -= 1;

	//2->1
	for (int i = 0; i < N_H_TLU; i++){
		float weighted_delta = 0.f;
		for (int j = 0; j < N_F_TLU; j++){
			weighted_delta += weight_layer1_to_layer2[i][j] * delta_layer_2[j];
		}
		delta_layer_1[i] = f_layer_1[i] * (1 - f_layer_1[i])*weighted_delta;
	}

	//1->0
	for (int i = 0; i < N_X_TLU; i++){
		float weighted_delta = 0.f;
		for (int j = 0; j < N_H_TLU; j++){
			weighted_delta += weight_layer0_to_layer1[i][j] * delta_layer_1[j];
		}
		delta_layer_0[i] = f_layer_0[i] * (1 - f_layer_0[i])*weighted_delta;
	}

	//===================================================================before
	//for (int l = N_LAYER - 1; l >= 0; l--){
	//   if (l != N_LAYER - 1){
	//      for (int i = 0; i < TLU[l]; i++){      // wdelta 는 l+1층의 델타 * l
	//         float weighted_delta = 0;
	//         for (int j = 0; j < TLU[l + 1]; j++){
	//            weighted_delta += delta[l + 1][j] * weight[l + 1][j][i];
	//         }
	//         delta[l][i] = f[l][i] * (1 - f[l][i])*weighted_delta;
	//      }
	//   }
	//   else{
	//      for (int i = 0; i < TLU[N_LAYER - 1]; i++){
	//         delta[N_LAYER - 1][i] = soft_max[i];
	//         if (i == goal - 1){
	//            delta[N_LAYER - 1][i] -= 1;
	//         }
	//      }
	//   }
	//}
}
//delta 를 이용해서 parameter 들을 갱신한다. bias, weight , word dictionary
void update(vector_window* window){
	int i = 0;
	for (; i < 50; i++){
		window->prev[i] -= LEARNING_RATE * delta_layer_0[i];
	}
	for (int j=0; i < 100; i++,j++){
		window->curr[j] -= LEARNING_RATE*delta_layer_0[i];
	}
	for (int j = 0; i < 150; i++, j++){
		window->next[j] -= LEARNING_RATE*delta_layer_0[i];
	}

	for (i = 0; i < N_X_TLU; i++){
		bias_layer_0[i] -= LEARNING_RATE*delta_layer_0[i];
		for (int j = 0; j < N_H_TLU; j++){
			weight_layer0_to_layer1[i][j] -= LEARNING_RATE * f_layer_0[i] * delta_layer_1[j];
		}
	}
	for (i = 0; i < N_H_TLU; i++){
		bias_layer_1[i] -= LEARNING_RATE*delta_layer_1[i];		
		for (int j = 0; j < N_F_TLU; j++){
			weight_layer1_to_layer2[i][j] -= LEARNING_RATE* f_layer_1[i] * delta_layer_2[j];
		}
	}
	for (i = 0; i < N_F_TLU; i++){
		bias_layer_2[i] -= LEARNING_RATE*delta_layer_2[i];
	}
	//====================================================================

	/* 사실 weighted delta를 할 필요가 없었다. 입력에서 0층에 대한 weight 는 0 이기 때문에 의미 0층의 델타를 이용해서 구하면 됬는데 좀 이상하게 구현했었음
	for (int i = 0; i < N_X_TLU; i++){
		float weighted_delta = 0;
		for (int j = 0; j < TLU[1]; j++){
			weighted_delta += delta[1][j] * weight[1][j][i];
		}
		if (i < 50){
			window->prev[i] -= LEARNING_RATE*weighted_delta;
		}
		else if (i < 100){
			window->curr[i - 50] -= LEARNING_RATE*weighted_delta;
		}
		else{
			window->next[i - 100] -= LEARNING_RATE*weighted_delta;
		}

		bias[0][i] -= LEARNING_RATE*delta[0][i];
	}

	for (int l = 1; l < N_LAYER; l++){
		for (int i = 0; i < TLU[l]; i++){
			bias[l][i] -= LEARNING_RATE*delta[l][i];

			for (int j = 0; j < TLU[l - 1]; j++){
				weight[l][i][j] -= LEARNING_RATE*delta[l][i] * f[l - 1][j];
			}
		}
	}*/

}
void file_read(NODE** root){   //   read origin_eng.txt and insert each word in tree      
	int line = 0;
	FILE* fp = fopen("word_origin_eng.txt", "r");
	char str[STR];
	char buf[WRD];
	int i = 0;
	while (!feof(fp)){
		line++;
		int i = 0, len = 0;
		fgets(str, STR, fp);
		len = strlen(str);
		while (str[0] != '='&&str[0] != '\n'&&i < len){
			sscanf(str + i, "%s", buf);
			insert(root, &buf[0]);
			i += strlen(buf) + 1;
		}
	}
	fclose(fp);
}
void numbering(NODE** root, dictionary** dict){            //   attach the number to each word and make word dictionary
	//FILE* fp = fopen("word.txt", "w"); //why error code ...?
	FILE* fp = fopen("word_.txt", "w");
	NODE* it = begin(*root);
	word_cnt = 1;         //first word -> num =1
	for (; it != end(*root); ToNext(&it), word_cnt++){
		it->num = word_cnt;
		fprintf(fp, "%10s\t:\t%10d\n", it->key, it->num);
	}
	word_cnt -= 1;
	fclose(fp);
	*dict = (dictionary*)calloc(word_cnt + 1, sizeof(dictionary));      // idx 1 ~ 51456(word_cnt)
	dictionary_genaerate(*dict);

}
void file_write(NODE* root, VECTOR** head, dictionary* dict){
	/*   make numbered text and classified text get train order                     */
	/*   word_orgin_eng.txt 와 word_origin_class.txt 는 단어의 갯수, 줄수가 동일하다.   */
	FILE* fp = fopen("word_origin_eng.txt", "r");
	FILE* w_fp = fopen("word_origin_num.txt", "w");
	FILE* class_fp = fopen("word_origin_class.txt", "r");
	FILE* train_fp = fopen("word_train_data.txt", "w");

	NODE* node;
	char str[STR] = { 0, };
	char class_str[STR] = { 0, };
	int word_class = { 0, };//1~45
	char buf[WRD] = { 0, };
	int len = 0;;
	data dummy;
	dummy.goal = dummy.word_num = 0;
	vec_create(head);
	vec_pushback(*head, dummy);

	/* eng 와 class 파일에서 한 줄씩 읽어서 eng를 이용해서 단어 번호를 찾고
	그 단어번호와 정답class를 매칭하는 작업을 한다.
	공백과 파일 번호를 포함한다.*/
	while (!feof(fp)){
		int num = -1;
		int i = 0;
		int j = 0;

		//파일 별로 한줄씩 읽는다
		fgets(class_str, STR, class_fp);
		fgets(str, STR, fp);

		//공백이면
		if (str[0] == '\n'){
			fputs(str, w_fp);
			fputs(str, train_fp);
		}
		//str이 시작이 ==== 이면 파일 번호를 의미한다. 
		else if (str[0] == '='&&str[1] == '='&&str[2] == '='&&str[3] == '='){
			int i = 0;
			char file_name[64] = { 0, };
			while (str[i] == '=')i++;
			sscanf(str + i, "%s", file_name);
			fprintf(w_fp, "# %s\n", file_name);
			fprintf(train_fp, "# %s", file_name);
		}
		// 일반 문장에 대한 처리부
		else{
			//str의 말단은 마지막 문장을 제외하고 '\n'이므로 이를 '\0'로 바꾼다.
			data data;
			str[strlen(str) - 1] = str[strlen(str) - 1] == '\n' ? '\0' : str[strlen(str) - 1];
			class_str[strlen(class_str) - 1] =
				class_str[strlen(class_str) - 1] == '\n' ? '\0' : class_str[strlen(class_str) - 1];

			len = strlen(str);
			while (*(str + i) != '\0'){
				sscanf(str + i, "%s", buf);
				sscanf(class_str + j, "%d", &word_class);

				node = Find(root, buf);
				if (node){
					num = node->num;
				}
				//order vector            
				data.word_num = num;
				data.goal = (dict + num)->goal = word_class;
				vec_pushback(*head, data);


				fprintf(w_fp, "%7d", num);
				fprintf(train_fp, "%16s%7d%3d\n", buf, num, word_class);
				i += strlen(buf) + 1;
				j += word_class > 9 ? 3 : 2;
			}
			fputc('\n', w_fp);
			fputc('\n', train_fp);
		}
	}
	vec_pushback(*head, dummy);
	fclose(fp);
	fclose(w_fp);
	fclose(class_fp);
	fclose(train_fp);
}
void epoch(VECTOR* head, dictionary* dict){      //head -> train order ,  dict -> word_vec
	int i = 1;
	int idx = 0;   //dictionary 
	int goal = 0;
	dictionary* curr = NULL;
	dictionary* prev = NULL;
	dictionary* next = NULL;
	vector_window v_window;
	int hit = 0;
	float precise = 0;

	// start i = 1  i(0) = dummy
	
	while (i < head->size*0.9){
	//while (i < 10000){

		goal = (head->element + i)->goal;
		prev = dict + (head->element + i - 1)->word_num;
		curr = dict + (head->element + i)->word_num;
		next = dict + (head->element + i + 1)->word_num;

		v_window.curr = curr->word_vec;
		v_window.next = next->word_vec;
		v_window.prev = prev->word_vec;


		//hit += 
		forward_computing(&v_window, goal);
		backward_computing(&v_window, goal, idx);
		update(&v_window);
		i++;
	}

	//printf("learning...\n");


}
double test(VECTOR* head, dictionary* dict){
	int i = (int)(head->size*0.9);
	int idx = 0;   //dictionary 
	int goal = 0;
	vector_window v_window;
	dictionary* curr = NULL;
	dictionary* prev = NULL;
	dictionary* next = NULL;
	int hit = 0;
	double precise = 0;
	while (i < head->size - 1){
		goal = (head->element + i)->goal;
		prev = dict + (head->element + i - 1)->word_num;
		curr = dict + (head->element + i)->word_num;
		next = dict + (head->element + i + 1)->word_num;

		v_window.curr = curr->word_vec;
		v_window.next = next->word_vec;
		v_window.prev = prev->word_vec;

		hit += forward_computing(&v_window, goal);
		i++;
	}

	precise = (double)hit / (head->size*0.1);
	return precise;
}