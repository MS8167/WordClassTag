#pragma warning (disable:4996)
#include<stdio.h>
#include<stdlib.h>
#include<time.h>	//rand
#include<math.h>	//exp	

#define N_L0	784
#define N_L1	512
#define N_L2	10
#define SIG(X) (1/(1+expf((-1*(X)))))
#define LEARNINGRATE	0.08f

float bias_0[N_L0];
float bias_1[N_L1];
float bias_2[N_L2];

float in_0[N_L0];
float in_1[N_L1];
float in_2[N_L2];

float out_0[N_L0];
float out_1[N_L1];
float out_2[N_L2];

float delta_0[N_L0];
float delta_1[N_L1];
float delta_2[N_L2];

float w01[N_L0][N_L1];
float w12[N_L1][N_L2];

float softmax[N_L2];



typedef struct IMG{
	unsigned char label;
	float img_data[784];//28*28=784
}IMG;
IMG train[60000] = { 0, };
IMG test[10000] = { 0, };
void mnist_data_ready(){
	FILE* train_image;
	FILE* train_label;	//train 은 6만
	FILE* test_image;
	FILE* test_label;	//test는 1만
	unsigned char c_data[784] = { 0, };
	int i = 0;

	unsigned char file_header[16] = { 0, };


	fopen_s(&train_image, "train-images.idx3-ubyte", "rb");
	fopen_s(&train_label, "train-labels.idx1-ubyte", "rb");

	fread(file_header, 1, 16, train_image);	//img 파일 헤더 16바이트
	fread(file_header, 1, 8, train_label);		//label 파일의 헤더 8 바이트


	while (!feof(train_image)){
		fread(&(train[i].label), 1, 1, train_label);	//label 데이터
		fread(c_data, 1, 784, train_image);
		for (int j = 0; j < 784; j++){
			if (c_data[j]!=0)	train[i].img_data[j] = (float)c_data[j]/255;
		}
		i++;
	}

	fopen_s(&test_image, "t10k-images.idx3-ubyte", "rb");
	fopen_s(&test_label, "t10k-labels.idx1-ubyte", "rb");

	fread(file_header, 1, 16, test_image);	//img 파일 헤더 16바이트
	fread(file_header, 1, 8, test_label);		//label 파일의 헤더 8 바이트

	i = 0;
	while (!feof(test_image)){
		fread(&(test[i].label), 1, 1, test_label);	//label 데이터
		fread(c_data, 1, 784, test_image);
		for (int j = 0; j < 784; j++){
			if (c_data[j] != 0)	test[i].img_data[j] = (float)c_data[j] / 255;
		}
		i++;
	}


	fclose(train_label);
	fclose(train_image);

	fclose(test_image);
	fclose(test_label);
}
void bias_generate(){
	int sign = 1;
	float rand_val = 0;
	srand((unsigned)time(NULL));

	for (int i = 0; i < N_L0; i++){
		rand_val = (float)rand() / RAND_MAX;
		sign = rand()>RAND_MAX / 2 ? -1 : 1;
		bias_0[i] = sign*rand_val;
	}
	for (int i = 0; i < N_L1; i++){
		rand_val = (float)rand() / RAND_MAX;
		sign = rand()>RAND_MAX / 2 ? -1 : 1;
		bias_1[i] = sign*rand_val;
	}
	for (int i = 0; i < N_L2; i++){
		rand_val = (float)rand() / RAND_MAX;
		sign = rand()>RAND_MAX / 2 ? -1 : 1;
		bias_2[i] = sign*rand_val;
	}
}
void weight_generate(){                     //   initialize weight (-1,1)
	int sign = 1;
	float rand_val = 0;
	srand((unsigned)time(NULL));

	for (int i = 0; i < N_L0; i++){
		for (int j = 0; j < N_L1; j++){
			rand_val = (float)rand() / RAND_MAX;
			sign = rand()>RAND_MAX / 2 ? -1 : 1;
			w01[i][j] = sign*rand_val;
		}
	}
	for (int i = 0; i < N_L1; i++){
		for (int j = 0; j < N_L2; j++){
			rand_val = (float)rand() / RAND_MAX;
			sign = rand()>RAND_MAX / 2 ? -1 : 1;
			w12[i][j] = sign*rand_val;
		}
	}
}
void do_train(int n){
	int i = 0, j = 0;
	float w_sum = 0.f;
	float w_delta = 0.f;
	float exp_sum = 0.f;

	//forward

	for (; i < N_L0; i++){
		out_0[i] = SIG(train[n].img_data[i] + bias_0[i]);
	}

	for (i = 0; i < N_L1; i++){
		w_sum = 0.f;
		for (j=0; j < N_L0; j++){
			w_sum += out_0[j] * w01[j][i];
		}
		in_1[i] = w_sum;
		out_1[i] = SIG(w_sum + bias_1[i]);
	}

	for (i = 0; i < N_L2; i++){
		w_sum = 0.f;
		for (j = 0; j < N_L1; j++){
			w_sum += out_1[j] * w12[j][i];
		}
		out_2[i]=in_2[i] = w_sum;
		exp_sum += softmax[i] = expf(out_2[i]);
	}

	for (i = 0; i < N_L2; i++){
		delta_2[i] = softmax[i] /= exp_sum;
	}
	delta_2[train[n].label] -= 1;
	
	//backward

	for (i = 0; i < N_L1; i++){
		w_delta = 0.f;
		for (j = 0; j < N_L2; j++){
			w_delta += delta_2[j] * w12[i][j];
		}
		delta_1[i] = out_1[i] * (1 - out_1[i])*w_delta;
	}

	for (i = 0; i < N_L0; i++){
		w_delta = 0.f;
		for (j = 0; j < N_L1; j++){
			w_delta += delta_1[j] * w01[i][j];
		}
		delta_0[i] = out_0[i] * (1 - out_0[i])*w_delta;
	}

	//update

	for (i = 0; i < N_L0; i++){
		bias_0[i] -= LEARNINGRATE*delta_0[i];
	}
	for (i = 0; i < N_L1; i++){
		bias_1[i] -= LEARNINGRATE*delta_1[i];
	}
	for (i = 0; i < N_L2; i++){
		bias_2[i] -= LEARNINGRATE*delta_2[i];
	}
	for (i = 0; i < N_L0; i++){
		for (j = 0; j < N_L1; j++){
			w01[i][j] -= LEARNINGRATE*delta_1[j] * out_0[i];
		}
	}

	for (i = 0; i < N_L1; i++){
		for (j = 0; j < N_L2; j++){
			w12[i][j] -= LEARNINGRATE*delta_2[j] * out_1[i];
		}
	}

}
int do_test(int n){
	int i = 0, j = 0;
	float w_sum = 0.f;
	float exp_sum = 0.f;
	int max_idx = 0;
	int hit = 0;
	//forward

	for (; i < N_L0; i++){
		out_0[i] = SIG(test[n].img_data[i] + bias_0[i]);
	}

	for (i = 0; i < N_L1; i++){
		w_sum = 0.f;
		for (j = 0; j < N_L0; j++){
			w_sum += out_0[j] * w01[j][i];
		}
		in_1[i] = w_sum;
		out_1[i] = SIG(w_sum + bias_1[i]);
	}

	for (i = 0; i < N_L2; i++){
		w_sum = 0.f;
		for (j = 0; j < N_L1; j++){
			w_sum += out_1[j] * w12[j][i];
		}
		out_2[i] = in_2[i] = w_sum;
		if (out_2[i]>out_2[max_idx])max_idx = i;
	}

	return hit = max_idx != (int)test[n].label ? 0 : 1;
}
void epoch(){
	int i = 0;
	int hit = 0;
	for (; i < 60000; i++){
		do_train(i);
	}
	for (i = 0; i < 10000; i++){
		hit += do_test(i);
	}
	printf("precise : %f", (float)hit / 10000);
}


int main(){
	double start = 0;
	mnist_data_ready();
	bias_generate();
	weight_generate();

	while (1){
		start = clock();
		epoch();
		start = clock() - start;
		printf("cpu epoch time : %d\n", start);
	}

	return 0;
}