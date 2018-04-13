#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <ctype.h>
#include <fcntl.h>

char* FILELIST = "./filelist.txt";
char* INPUTLIST = "./inputlist.txt";
char* DESTINATION = "../";
int BUF_SIZE = 128;

void ls() {
	char* argv[] = {"/bin/ls", NULL};
	char* env[] = {NULL};
	FILE* fp = fopen(FILELIST, "w");
	fclose(fopen(INPUTLIST, "w"));
	int status;
	pid_t child_pid;

	dup2(fileno(fp), 1);
	if ((child_pid = fork()) == 0) {
		if (execve(argv[0], argv, env) < 0) {
			printf("Error\n");
			exit(0);
		}
	}
	waitpid(child_pid, &status, 0);
	fclose(fp);

	return;
}

int lst_num() {
	FILE* fp_filelist = fopen(FILELIST, "r");
	char buf[BUF_SIZE];
	int result = 0;
	while (fgets(buf, BUF_SIZE, fp_filelist)) result++;
	fclose(fp_filelist);
	return result - 4;
}

int* rand_arr(int n) {
	int* p = (int *) malloc(n * sizeof(int));
	for (int i = 0; i != n; i++) p[i] = i;
	srand(time(NULL));
	for (size_t i = 0; i != n - 1; i++) {
		size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
		int t = p[j];
		p[j] = p[i];
		p[i] = t;
	}
	return p;
}

int slct(int* idx, int n, int m) {
	FILE* fp_filelist = fopen(FILELIST, "r");
	FILE* fp_inputlist = fopen(INPUTLIST, "w");
	char** buf = (char **) malloc(n * sizeof(char*));
	for (int i = 0; i != n; i++) buf[i] = (char *) malloc(BUF_SIZE * sizeof(char));
	for (int i = 0; i != n; i++) fgets(buf[idx[i]], BUF_SIZE, fp_filelist);
	int length = n < m ? n : m;
	for (int i = 0; i != length; i++) fputs(buf[i], fp_inputlist);
	free(idx);
	for (int i = 0; i != n; i++) free(buf[i]);
	free(buf);
	fclose(fp_filelist);
	fclose(fp_inputlist);

	return length;
}

void mv(int length) {
	char* argv[] = {"/bin/mv", NULL, DESTINATION, NULL};
	char* env[] = {NULL};
	FILE* fp = fopen(INPUTLIST, "r");
	int status;
	char** buf = (char **) malloc(length * sizeof(char*));
	for (int i = 0; i != length; i++) buf[i] = (char *) malloc(BUF_SIZE * sizeof(char));

	for (int i = 0; i != length; i++) {
		fgets(buf[i], BUF_SIZE, fp);
		int len = strlen(buf[i]);
		if (len > 0 && buf[i][len - 1] == '\n') buf[i][len - 1] = '\0';
	}
	fclose(fp);

	for (int i = 0; i != length; i++) {
		if (fork() == 0) {
			argv[1] = buf[i];
			if (execve(argv[0], argv, env) < 0) {
				printf("Error\n");
				exit(0);
			}
		}
	}
	for (int i = 0; i != length; i++) wait(&status);

	for (int i = 0; i != length; i++) free(buf[i]);
	free(buf);

	return;
}

int main(int argc, char* argv[]) {
	if (argc != 2 || !isdigit(argv[1][0])) {
		printf("Argument Error\n");
		return 0;
	}
	int num_arg = atoi(argv[1]);
	ls();
	int num_lst = lst_num();
	int length = slct(rand_arr(num_lst), num_lst, num_arg);
	mv(length);
	return 0;
}