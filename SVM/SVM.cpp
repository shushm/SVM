// SVM.cpp : �������̨Ӧ�ó������ڵ㡣
//-------------------------------------------------------//
//��������֧��������RBF��ѵ�����Գ���
//���ߣ�����
//���°汾��1.1 alpha ����˵�� �޸�1.0beta�汾��������������
//����޸�ʱ�䣺2015-5-28||13��07
//֮ǰ�汾:
//--------------------------------------------------------//
//1.0 beta ����˵�� �����Զ���֤��ѵ�����ﵽԤ��׼ȷ�ʽ��
//1.0 alpha         �������Բ�׼ȷ����
//-------------------------------------------------------//
#include "stdafx.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <time.h>
#include <windows.h>

using namespace std;

int LineCount(char *filename);

double **x;
double *y;
double *alpha;
double **dotCache;
double **kernel;
double *errorCache;
double b = 0.0;
double eps = 0.001;
/*
double x[2000][5000] = { 0 };		//��������
double y[2000];					//������ǩ

double b = 0.0;
double alpha[2000] = { 0 };	//�������ճ���


double dotCache[2000][2000];
//�˻���
double kernel[2000][2000];
double errorCache[2000];		//����
double eps = 0.001;			//��ֹ�����Ĳ�ֵ 
*/
//
// rbf kernel for exp(-gamma*|u-v|^2) ��˹������˺������ޱ��Σ�
//


//
//	ѵ���ò���
//

/*
//�������
double dotCache[m][m];
//�˻���
double kernel[m][m];


//
double x[m][n] = { 0 };		//��������
double y[m];					//������ǩ

double b = 0.0;
double alpha[m] = { 0 };	//�������ճ���
double C = 5;			//�ͷ�����
double tolerance = 0.001;	//�ɳڱ���

double errorCache[m];		//����
double eps = 0.001;			//��ֹ�����Ĳ�ֵ 

//
// rbf kernel for exp(-gamma*|u-v|^2) ��˹������˺������ޱ��Σ�
//
double sigma =10; //
double gamma = 1/(sigma*sigma);

//
//	ѵ���ò���
//
int maxIter =5000; //��������������
int iterCount = 0; //��������
int numChanged = 0;
bool examineAll = true;

//�������
double dotCache[m][m];
//�˻���
double kernel[m][m];

*/

double dot(double x[], int xLen, double y[],int yLen)
{
	double sum=0;
	int i=0, j = 0;
	while (i < xLen&&j < yLen)
	{
		sum+=x[i] * y[j];   //��������ά���̶� �����̶�����Ҫ�Ƚϳ��� 
		i++;
		j++;
	}
	return sum;
}

//
//ѵ��ʱ�õĺ˺���
//
double kernelFunction(int i1, int i2,double gamma)
{
	double result = 0.0;
	result = exp(-gamma * (dotCache[i1][i1] + dotCache[i2][i2] - 2 * dotCache[i1][i2])); 
	return result;
}
//
//
//
double dataInit(int M,int N,double gamma) //����������M    ά����N
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			dotCache[i][j] = dot(x[i],N,x[j],N);
		}	
	}

	for (int i = 0; i < M; i++)
	{
		if (kernel[i] == NULL)
		{
			cout << "�ڴ�ľ�" << endl;
			exit(1);
		}
		for (int j = 0; j < M; j++)
		{
			kernel[i][j] = kernelFunction(i, j,gamma);
		}
	}
	return 1;
}

double learnFunc(int k,int SampleNum)  //ѧϰ����U ,�����
{
	double sum = 0;
	for (int i = 0; i < SampleNum; i++)
	{
		sum += alpha[i] * y[i] * kernel[i][k];
	}
	sum = sum + b;
	return sum;
}
double calError(int k, int SampleNum)
{
	double error = learnFunc(k, SampleNum) - y[k];
	return error;

}

double updateErrorCache(int k, int SampleNum)
{
	double error = calError(k, SampleNum);
	errorCache[k] = error;

	return 1;
}

int selectMaxJ(double E1,int SampleNum,double C)
{
	int i2 = -1;
	double tmax = 0.0;
	for (int k = 0; k < SampleNum; k++)
	{
		if (0 < alpha[k] && alpha[k] < C)
		{
			double E2 = errorCache[k];
			double tmp = abs(E2-E1);
			if (tmp>tmax)
			{
				tmax = tmp;
				i2 = k;
			}
		}
	}
	return i2;
}
//
//���������,���ѡ��i1�������ܵ���i2
//
int randomSelect(int i1,int  SampleNum)
{
	int i2 = 0;
	do
	{
		srand((int)time(0));
		i2 = rand() % SampleNum;  //
	} while (i2 == i1);

	return i2;
}
/**
*  ��������
*  input i1,i2
*  return 0/1
*/ 
int takeStep(int i1, int i2, int SampleNum,double C)
{

	if (i1 == i2) return 0;
	double alpha1 = alpha[i1];
	double alpha2 = alpha[i2];
	double y1 = y[i1];
	double y2 = y[i2];
	double E1 = 0;
	double E2 = 0;
	double a1, a2;
	double s = y1*y2;
	//double L,H
	double L, H;
	//compute E1
	if (0 < alpha1 && alpha1 < C)
		E1 = errorCache[i1];
	else
		E1 = calError(i1, SampleNum);
	//compute E2
	if (0 < alpha2 && alpha2 < C)
		E2 = errorCache[i2];
	else
		E2 = calError(i2, SampleNum);
	//compute L,H
	if (y1 == y2)
	{
		/*double aa = alpha1 + alpha2;
		if (aa>C)
		{
			L = aa - C;
			H = C;
		}
		else
		{
			L = 0;
			H = aa;
		}*/
			L = fmax(0,alpha1+alpha2-C);
			H = fmin(C,alpha1+alpha2);
	}
	else{
		/*double aa = alpha1 - alpha2;
		if (aa > 0)
		{
			L = 0;
			H = C - aa;
		}
		else
		{
			L = -aa;
			H = C;
		}*/
			L = fmax(0,alpha2-alpha1);
			H = fmin(C, C + alpha2 - alpha1);

	}
	if (L >= H)
		return 0;

	double k11 = kernel[i1][i1];
	double k12 = kernel[i1][i2];
	double k22 = kernel[i2][i2];


	double eta = 2 * k12 - k11 - k22;

	if (eta < 0)
	{
		a2 = alpha2 - y2*(E1 - E2) / eta;
		if (a2 < L)
			a2 = L;
		else if (a2>H)
			a2 = H;

	}
	else
	{
		double C1 = eta / 2.0;
		double C2 = y2*(E1 - E2) - eta*alpha2;
		double Lobj = C1*L*L + C2*L;
		double Hobj = C1*H*H + C2*H;
		if (Lobj > Hobj + eps)
			a2 = L;
		else if (Lobj < Hobj - eps)
			a2 = H;
		else
			a2 = alpha2;
	}

	if (fabs(a2 - alpha2) < eps*(a2 + alpha2 + eps))
		return 0;

	//Update a1
	a1 = alpha1 + s*(alpha2 - a2);
	
	if (a1 < 0)
	{
		a2 += s*a1;
		a1 = 0;
	}
	else if (a1>C)
	{
		a2 += s*(a1 - C);
		a1 = C;
	}

	double b1 = b -E1 -y1*(a1 - alpha1)*k11 - y2*(a2 - alpha2)*k12;
	double b2 = b- E2 -y1*(a1 - alpha1)*k12 - y2*(a2 - alpha2)*k22;

	double bNew = 0;
	
	if (0 < a1 && a1 < C)
		bNew = b1;
	else if (0 < a2 && a2 < C)
		bNew = b2;
	else
		bNew = (b1 + b2) / 2;

	//double deltaB = bNew - b;
	b = bNew;
	//
	//��������
	//
	//double t1 = y1*(a1 - alpha1);
	//double t2 = y2*(a2-alpha2);
	//for (int i = 0; i < m; i++)
	//{
	//	if (0 < alpha[i] && alpha[i] < C)
	//	{
	//		errorCache[i] += t1*kernel[i1][i]+t2*kernel[i2][i]-deltaB;
	//
	//	}

	//}
	//errorCache[i1] = 0;
	//errorCache[i2] = 0;

	updateErrorCache(i1, SampleNum);
	updateErrorCache(i2, SampleNum);

	alpha[i1] = a1;
	alpha[i2] = a2;

	return 1;
}
//
//Ԥ���ú˺���  ���� gamma=0.5
//
double kFunction(double x[], double y[], double gamma, int Diamension)
{
	double sum = 0;
	int i = 0, j = 0;
	while (i<Diamension || j<Diamension)
	{
		if (i== j)
		{
			double d = x[i] - y[j];
			sum += d*d;
			i++;
			j++;
		}
		/*else if (i>j)
		{
			sum += x[i]*x[i];
			j++;
		}
		else
		{
			sum += x[i] * x[i];
			i++;
		}*/
	}
	return exp(-gamma*sum);   //��˹��
}

double examineExample(int i1, int SampleNum,double C,double gamma,double tolerance)
{
	double y1 = y[i1];
	double alpha1 = alpha[i1];
	double E1 = 0;

	if (0 < alpha1&&alpha1 < C)
		E1 = errorCache[i1];
	else
		E1 = calError(i1, SampleNum);

	//printf(" E1  =  %f\n",E1);
	double r1 = E1*y1;  
	//printf(" r1  =  %f\n", r1);
	if (((r1<-tolerance) && alpha1<C) || (r1>tolerance&&alpha1>0))
	{
		//ѡ��E1-E2�����ĵ�
		int i2 = selectMaxJ(E1, SampleNum,C);
		if (i2 >= 0)
		{
			if (takeStep(i1, i2, SampleNum,C))
				return 1;
		}
		//ѡ��0<alpha<C�ĵ����ѡȡһ����ʼ��
		int k0 = randomSelect(i1, SampleNum);
		//printf(" k0  =  %d \n", k0);
		for (int k = k0; k < SampleNum + k0; k++)
		{
			i2 = k%SampleNum;
			//printf(" i1  =  %d \n", i1);
			if (0<alpha[i2]&&alpha[i2]<C)
			{
				if (takeStep(i1, i2, SampleNum,C))
					return 1;
			}

		}

		//��������ϣ��ٱ���ȫ����
		k0 = randomSelect(i1, SampleNum);
		//printf(" k0  =  %d \n", k0);
		for (int k = k0; k < SampleNum + k0; k++)
		{
			i2 = k%SampleNum;
			if (takeStep(i1, i2, SampleNum,C))
				return 1;
			

		}

	}

	return 0;
}

//testfile1 ���������� 
//testfile2 ����������

double predict(char *testfile1,char *testfile2, int  SampleNum, int Diamension,double gamma)
{
	double probablity = 0.0;
	int correctCount = 0;
	
	int file1num = 0;
	int file2num = 0;
	file1num = LineCount(testfile1); //ͳ�Ʋ�����������
	file2num = LineCount(testfile2);
	int tn = file1num+file2num; //������������
	std::cout << "-------------------------------------------------" << endl;
	std::cout << "��������������" << tn << endl;
	//�򿪲�������1
	
	double *TestData;
	double **tx1;
	double **tx2;
	double *ty1;
	double *ty2;
	//----------------�����ļ�1----------------------------//
	ifstream file1(testfile1);
	//�����ڴ�
	TestData = new double[file1num*Diamension];
	tx1 = new double*[file1num];
	for (int i = 0; i < file1num; i++)
	{
		tx1[i] = new double[Diamension];
	}
	//tx2 = new double*[file2num*Diamension];
	ty1 = new double[file1num];
	//tx1 = new double[file1num*(Diamension + 1)];
	for (int i = 0; i < file1num*(Diamension); i++)
	{
		file1 >> TestData[i];
	}
	for (int i = 0; i < file1num; i++)
	{
		for (int j = 0; j < Diamension; j++)
		{
			tx1[i][j] = TestData[j + Diamension*i];
		}
		ty1[i] = 1;
	}
	//
	delete TestData;

	for (int i = 0; i < file1num; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < SampleNum; j++)
		{
			sum += alpha[j] * y[j] * kFunction(x[j], tx1[i], gamma, Diamension); //��֤ ���ߺ���
		}
		sum += b;
		if (sum>0 )//&& ty1[i]>0
			correctCount++;
	}
	//
	delete []ty1;
	for (int i = 0; i < file1num; i++)
	{
		delete [] tx1[i];
	}
	delete [] tx1;
	tx1 = NULL;
	file1.close();
	//----------------�����ļ�2----------------------------//
	ifstream file2(testfile2);
	//�����ڴ�
	TestData = new double[file2num*Diamension];
	tx2 = new double*[file2num];
	for (int i = 0; i < file2num; i++)
	{
		tx2[i] = new double[Diamension];
	}
	//tx2 = new double*[file2num*Diamension];
	ty2 = new double[file2num];
	//tx1 = new double[file1num*(Diamension + 1)];
	for (int i = 0; i < file2num*(Diamension); i++)
	{
		file2 >> TestData[i];
	}
	for (int i = 0; i < file2num; i++)
	{
		for (int j = 0; j < Diamension; j++)
		{
			tx2[i][j] = TestData[j + Diamension*i];
		}
		ty2[i] = -1;
	}
	//
	delete TestData;

	for (int i = 0; i < file2num; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < SampleNum; j++)
		{
			sum += alpha[j] * y[j] * kFunction(x[j], tx2[i], gamma, Diamension); //��֤ ���ߺ���
		}
		sum += b;
		if (sum < 0 )//&&ty2[i] < 0
			correctCount++;
	}
	//
	delete[] ty2;
	for (int i = 0; i < file2num; i++)
	{
		delete[] tx2[i];
	}
	delete[] tx2;
	tx2 = NULL;
	file2.close();
	ofstream log;
	log.open("log.txt", ios::app | ios::out);
	//------------------------------------------//
	probablity = (double) correctCount / tn;

	log << "TestNum="<<tn<< endl;
	log << "correctCount=" << correctCount << "; Probablity=" << probablity*100.0<<"%" << endl;
	log << "-----------Test End-----------------------"<< endl;
	log.close();
	std::cout << "correctCount=" << correctCount << "; Probablity=" << probablity*100.0 << "%" << endl;
	return probablity;

}

int _tmain(int argc, _TCHAR* argv[])
{

	double * DataCache;
	//ѵ����ز���
	double sigma = 5; //
	double gamma = 1 / (sigma*sigma);
	double C = 4;			//�ͷ�����
	double tolerance = 0.001;	//�ɳڱ���
	//
	int SampleNum = 0;  //��������
	int Diamension =3780;  //����ά�� Ĭ��3780hog����ά��
	//
	int maxIter = 50000; //��������������
	int iterCount = 0; //��������
	int numChanged = 0;
	bool examineAll = true;
	//
	ofstream log;
	log.open("log.txt", ios::app | ios::out);
	//
	//log << "The SVM Train and Test Program Log file" << endl;
	log << "-----------------------------------------------" << endl;
	log.close();
	//
	std::cout << "-******************************************************-" << endl;
	std::cout << "---------------A Simple SVM Program---------------------" << endl;
	std::cout << "----------HuamingShen@Shanghai University---------------" << endl;
	std::cout << "-------------Use the RBF kernel function----------------" << endl;
	std::cout << "--------------------V1.1 beta---------------------------" << endl;
	std::cout << "--------------Thanks to John Platt----------------------" << endl;
	std::cout << "-******************************************************-" << endl;
	std::cout << "---------------------����ѡ��---------------------------" << endl;
	std::cout << "-               help          ----����ѡ��             -" << endl;
	std::cout << "-               Train         ----ѵ��ѡ��             -" << endl;
	std::cout << "-               Test          ----����ѡ��             -" << endl;
	std::cout << "-               Auto          ----�Զ�ѵ��ѡ��         -" << endl;
	std::cout << "-               quit          ----�˳�����             -" << endl;
	std::cout << "--------------------------------------------------------" << endl;
	std::cout << "-                   ʹ��˵��                           -" << endl;
	std::cout << "-               ��һ��      ѵ��                       -" << endl;
	std::cout << "-               �ڶ���--���Ի���ѵ��                   -" << endl;
	std::cout << "-               ѵ��ģ��׷����model�ļ�                -" << endl;
	std::cout << "-               ���Լ�¼׷����log�ļ���                -" << endl;
	std::cout << "--------------------------------------------------------" << endl;
	std::cout << "Enjoy yourself!" <<endl<< endl;
	std::cout << "��������:";
	 char choice[5];
	cin >> choice;
	//ѵ��������
	char PossamFile[100]="p1.dat";
	char NegsamFile[100]="n1.dat";
	//
	//FirstWrite = 1;
	//ѵ���õ����� �ڴ��Ƿ������־��һ��������ˣ��ٴ�����ǰ��Ҫ�ͷ��ڴ�
	char RamApply = 0;
	//
	while (1)
	{
		if ((_strnicmp(choice,"help",4))==0)
		{
			memset(choice, 0, 5);
			std::cout << "<<------------------����ѡ��------------------->>" << endl;
			std::cout << "               help          ----����ѡ��        " << endl;
			std::cout << "               Train         ----ѵ��ѡ��        " << endl;
			std::cout << "               Test          ----����ѡ��        " << endl;
			std::cout << "               Auto          ----�Զ�ѵ��ѡ��    " << endl;
			std::cout << "               quit          ----�˳�����        " << endl;
			std::cout << "-------------------------------------------------" << endl;

			std::cout << "��������:";
		}
		else if ((_strnicmp(choice, "Train", 4)) == 0)
		{
			memset(choice, 0, 5);
			if (RamApply == 1)
			{
				RamApply = 0;
				delete[] y;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] x[i];
				}
				delete[] x;
				delete[] alpha;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] dotCache[i];
				}
				delete[] dotCache;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] kernel[i];
				}
				delete[] kernel;
				delete[] errorCache;

			}
			/*
			//�������
			for (int i = 0; i < 2000; i++)
			{
				memset(x[i], 0, 5000);
			}
			for (int i = 0; i < 2000; i++)
			{
				memset(dotCache[i], 0, 2000);
			}
			for (int i = 0; i < 2000; i++)
			{
				memset(kernel[i], 0, 2000);
			}
			memset(alpha, 0, 2000);
			memset(errorCache, 0, 2000);//
			*/
			//���b
			b = 0;
			iterCount = 0; //��������
			numChanged = 0;
			examineAll = true;
			//
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "�˺��� RBF--> exp((-||x-x'||^2)/(sigma^2))" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "Ĭ�ϵĲ���Ϊ>>>>>" << endl;
			std::cout << "C=" << C << "; sigma=" << sigma << endl;
			std::cout << "���ѭ������=" << maxIter << "; �ɳڱ���=" << tolerance << endl;
			std::cout << "��ֹ��������=" << eps << endl;
			//
			char ic;
			std::cout << "�Ƿ�Ҫ���Ĳ�������Y/N��";
			cin >> ic;
			if (ic == 'Y' || ic == 'y')
			{
				std::cout << "������ͷ�����C=";
				cin >> C;
				std::cout << "���������sigma=";
				std::cin >> sigma;
				std::cout << "�ɳڱ���=";
				cin >> tolerance;
				std::cout << "���ѭ������=";
				cin >> maxIter;
				std::cout << "��ֹ��������=";
				cin >> eps;
			}
			std::cout << "---------------------------------------------" << endl;

			//��������

			//����ά�� hog:3780
			cout << "ά��=" << Diamension << endl;
			char dc;
			cout << "�Ƿ���Ҫ�ı�����ά����(Y/N)";
			cin >> dc;
			if (dc == 'Y' || dc == 'y')
			{
				std::cout << "����������ά����";
				cin >> Diamension;
			}

			//cout <<"����ά��Ϊ��" <<Diamension << endl;
			cout <<"ά��="<<Diamension << endl;
			std::cout << "����ѵ���������ļ���";
			cin >> PossamFile;
			ifstream fin(PossamFile);
			while (!fin)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "����������ѵ���������ļ���";
				cin >> PossamFile;
				ifstream fin(PossamFile);
				if (fin)
					break;
				//break;
				//return 0;
			//	std::cout << "����ѵ���������ļ���";
			//	cin >> PossamFile;

			}
			fin.close();
			std::cout << "����ѵ���������ļ���";
			cin >> NegsamFile;
			ifstream fin2(NegsamFile);
			if (!fin2)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "����������ѵ���������ļ���";
				cin >> NegsamFile;
				ifstream fin2(NegsamFile);
				if (fin2)
					break;
			}
			fin2.close();
			std::cout << "�����ļ�����" << endl;
			std::cout << "׼����ʼ>>>>>�ȴ�����ʱ���Գ�>>>>>>>>>" << endl;


			//��¼��ʼʱ��
			long begintime = GetTickCount();


			//cout << "����ʱ�䣺"<<ctime(&timeval) << endl;

			int PosNum = LineCount(PossamFile); //��������
			int NegNum = LineCount(NegsamFile); //��������
			SampleNum = PosNum + NegNum;  //��������
			std::cout << "��������= " << SampleNum << endl;
			std::cout << "����ά��= " << Diamension << endl;
			//��ʼΪ��������ռ�
			x = new double*[SampleNum];
			for (int i = 0; i < SampleNum; i++)
			{
				x[i] = new double[Diamension];
			}
			y = new double[SampleNum];
			alpha = new double[SampleNum];
			dotCache = new double*[SampleNum];
			for (int i = 0; i < SampleNum; i++)
			{
				dotCache[i] = new double[SampleNum];
			}

			kernel = new double*[SampleNum];
			for (int i = 0; i < SampleNum; i++)
			{
				kernel[i] = new double[SampleNum];
				if (kernel[i] == NULL)
				{
					cout << "�ڴ�ľ�"<< endl;
					exit(1);
				}
			}
			errorCache = new double[SampleNum];
			b = 0;
			for (int i = 0; i < SampleNum; i++)
			{
				alpha[i] = 0;
				errorCache[i] = 0;
			}
			
			//���ram�Ѿ����뵽
			RamApply = 1;

			//
			ofstream ofile;
			//��־�ļ�

			//�ļ��� Ŀǰ�û������޸�
			ofile.open("model.txt",ios::out|ios::app); //���ģ���ļ�
			//

			//log.se;
			//ģ�����ݶ��ļ����
			ofile << "----------------------------------------------------" << endl;
			ofile << "----------------The SVM model:---------------------- " << endl;
			ofile << "----------------------------------------------------" << endl;
			//��Ļ��ʾ��ʼѵ��
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "Train Begin ..." << endl;
			std::cout << "---->>>>>>>>>>>>>>>" << endl;
			std::cout << "�˺���ѡȡ-��˹��" << endl;
			std::cout << "C=" << C << "; sigma=" << sigma << endl;
			std::cout << "���ѭ������=" << maxIter << "; �ɳڱ���=" << tolerance << endl;
			std::cout << "��ֹ��������=" << eps << endl;
			//
			ofile << "Kernel: RBF" << endl;
			//
			std::cout << "Loading Data---->>>>>>>>>>>>>" << endl;
			//���ļ�
			//
			ofile << "gamma = " << gamma << endl;
			ofile << "C = " << C << endl;
			ofile << "ѵ������=" << PossamFile << ";" << NegsamFile << endl;

			//��ȡ�ļ�
			ifstream posfile(PossamFile);
			ifstream negfile(NegsamFile);

			//�����ڴ�
			DataCache = new double[SampleNum*Diamension];
			//��������
			for (int i = 0; i < SampleNum*(Diamension); i++)
			{
				if (i<PosNum*Diamension)
					posfile >> DataCache[i];
				else
					negfile >> DataCache[i];
			}
			//��������
			for (int i = 0; i < SampleNum; i++)
			{
				for (int j = 0; j < Diamension; j++)
				{
					x[i][j] = DataCache[j + (Diamension)*(i)];
				}
				if (i < PosNum)
					y[i] = 1;
				else
					y[i] = -1;
			}
			//�ͷ��ڴ�
			delete [] DataCache;

			posfile.close();
			negfile.close();



			std::printf("Load Success\nBegin to Intialize data\n");

			dataInit(SampleNum, Diamension, gamma);
			std::printf("Intialize OK \n");
			std::cout << "Training............"<< endl;
			while ((iterCount<maxIter) && (numChanged>0 || examineAll))
			{
				numChanged = 0;
				if (examineAll)
				{
					for (int i = 0; i < SampleNum; i++)
					{
						if (examineExample(i, SampleNum, C, gamma, tolerance))
							numChanged++;
						//printf("examineAll=%d \n", examineAll);
						//printf("numChanged=%d \n", numChanged);
					}
				}
				else
				{
					for (int i = 0; i < SampleNum; i++)
					{
						if (alpha[i] != 0 && alpha[i] != C) //alpha[i] != 0 && alpha[i] != C
						{
							if (examineExample(i, SampleNum, C, gamma, tolerance))
								numChanged++;
							//printf("examineAll=%d\n", examineAll);
							//printf("numChanged=%d \n", numChanged);
						}
					}
				}
				//printf("numChanged=%d \n", numChanged);
				iterCount++;
				//printf("iterCount =  %d \n", iterCount);
				if (examineAll)
					examineAll = false;
				else if (numChanged == 0)
					examineAll = true;

			}

			std::printf("Train Success\n");
			std::cout << "ѵ���ܼ�ѭ������ :" << iterCount << "��" << endl;
			long endtime = GetTickCount();
			//cout << "����ʱ�䣺" << ctime(&timeval2) << endl;
			long diftime = endtime - begintime;
			std::cout << "ѵ���ܼ�ʱ�� :" << (double)diftime / 1000 << "��" << endl;

			int Nsv = 0;
			for (int i = 0; i < SampleNum; i++)
			{
				if (alpha[i]>0)
					Nsv++;       //ͳ��֧����������
			}
			std::cout << "Number of Support Vector=" << Nsv << endl;
			ofile << "Number of Support Vector=" << Nsv << endl;
			ofile << "----------------------------------------------------" << endl;
			for (int i = 0; i < SampleNum; i++)
			{
				//printf("alpha %d =  %f \n",i, alpha[i]);
				if (alpha[i]!=0)
					ofile << i << " Label=" << y[i] << ";" << " alpha" << i << " : " << alpha[i] << endl;//<<" " <<x[i][0] <<" "<<x[i][1]
			}
			std::cout << "���ӽ����ģ���ļ���" << endl;
			std::printf("b =  %f \n", b);
			ofile << "b= " << b << endl;
			ofile.close();
			std::cout << "-----------------------------------------------------" << endl;
			std::cout << "��������:";
		}
		else if ((_strnicmp(choice, "test", 4)) == 0)
		{
			memset(choice, 0, 5);
			//�������ݼ�¼
			log.open("Testlog.txt",ios::app|ios::out);
			log << "-----------Test Start-----------------------" << endl;
			log << "C=" << C << " ;sigma=" << sigma << " ;ѭ������ =" << maxIter << endl;
			log << "eps=" << eps << ";tolerence=" << tolerance << endl;
			log << "ѵ������=" << PossamFile << ";" << NegsamFile << endl;
			//������ʾ����
			std::cout << ">>>>��ʼ����-------->>>>>>>>>" << endl;
			std::cout << "----------------------------------" << endl;
			std::cout << "У��ѵ������---->>>>>"<<endl;
			std::cout << "C=" << C << " ;sigma=" << sigma << " ;ѭ������ =" << maxIter << endl;
			std::cout << "eps=" << eps << ";tolerence=" << tolerance << endl;
			std::cout << "ѵ������=" << PossamFile << ";" << NegsamFile << endl;
			std::cout << "----------------------------------" << endl;
			
			std::cout << "�������������:";
			char TestPosfile[200];
			cin >> TestPosfile;

			ifstream fin(TestPosfile);
			while (!fin)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "��������������������ļ���";
				cin >> TestPosfile;
				ifstream fin(TestPosfile);
				if (fin)
					break;
			}
			fin.close();


			char TestNegfile[200];
			std::cout << "������Ը�����:";
			cin >> TestNegfile;

			ifstream fin2(TestNegfile);
			while (!fin2)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "������������Ը������ļ���";
				cin >> TestNegfile;
				ifstream fin2(TestNegfile);
				if (fin2)
					break;
			}
			fin2.close();

			log << "���������ļ�=" << TestPosfile << ";" <<TestNegfile << endl;
			std::cout << "����ļ��ϴ󣬻���ʱ��ϳ��������ĵȴ�>>>>>>>>>" << endl;
			predict(TestPosfile, TestNegfile, SampleNum, Diamension, gamma);
			std::cout << "-----------------------------------------------------" << endl;
			std::cout << "��������:";
			log.close();
		}
		else if ((_strnicmp(choice, "quit",4))==0)
		{
			memset(choice, 0, 5);
			if (RamApply == 1)
			{
				RamApply = 0;
				delete[] y;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] x[i];
				}
				delete[] x;
				delete[] alpha;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] dotCache[i];
				}
				delete[] dotCache;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] kernel[i];
				}
				delete[] kernel;
				delete[] errorCache;

			}
			std::cout << ">>>>>>>>ллʹ��>>>>>>>>>>>>>" << endl;
			//Sleep(3 * 1000);
			return 0;
		}//-------------------------------�Զ�ѵ������-------------------------------//
		else if ((_strnicmp(choice, "auto", 4)) == 0)//--------------------------------------------------------//
		{
			memset(choice, 0, 5);
			std::cout << "��ѡ���Զ�ѵ��>>>>>>>>>>>>>>>" << endl;
			//�Ѿ������X Y���ڴ� ��Ҫ�ͷ�������
			if (RamApply == 1)
			{
				RamApply = 0;
				delete[] y;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] x[i];
				}
				delete[] x;
				delete[] alpha;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] dotCache[i];
				}
				delete[] dotCache;
				for (int i = 0; i < SampleNum; i++)
				{
					delete[] kernel[i];
				}
				delete[] kernel;
				delete[] errorCache;

			}//-----------------------------------------------------------------------------//
			//ÿһ�ν��붼��Ҫ�������
			/*
			for (int i = 0; i < 2000; i++)
			{
				memset(x[i], 0, 5000);
			}
			for (int i = 0; i < 2000; i++)
			{
				memset(dotCache[i], 0, 2000);
			}
			for (int i = 0; i < 2000; i++)
			{
				memset(kernel[i], 0, 2000);
			}
			memset(alpha, 0, 2000);
			memset(errorCache, 0, 2000);//
			*/
			//���b
			b = 0;
			iterCount = 0; //��������
			numChanged = 0;
			examineAll = true;
			//
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "�˺��� RBF--> exp((-||x-x'||^2)/(sigma^2))" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "Ĭ�ϵĲ���Ϊ>>>>>" << endl;
			std::cout << "C=" << C << "; sigma=" << sigma << endl;
			std::cout << "���ѭ������=" << maxIter << "; �ɳڱ���=" << tolerance << endl;
			std::cout << "��ֹ��������=" << eps << endl;
			//
			char ic;
			std::cout << "�Ƿ�Ҫ����ѵ����������Y/N��";
			cin >> ic;
			if (ic == 'Y' || ic == 'y')
			{
				std::cout << "������ͷ�����C=";
				cin >> C;
				std::cout << "���������sigma=";
				std::cin >> sigma;
				std::cout << "�ɳڱ���=";
				cin >> tolerance;
				std::cout << "���ѭ������=";
				cin >> maxIter;
				std::cout << "��ֹ��������=";
				cin >> eps;
			}
			std::cout << "---------------------------------------------" << endl;

			//����ά�� hog:3780
			cout << "ά��=" << Diamension << endl;
			char dc=0;
			cout << "�Ƿ���Ҫ�ı�����ά����(Y/N)";
			cin >> dc;
			if (dc == 'Y' || dc == 'y')
			{
				std::cout << "����������ά����";
				cin >> Diamension;
				cout << "ά��=" << Diamension << endl;
				dc = 0;
			}

			
			cout << "-------------------------------------"<< endl;
			cout << "Ĭ��ѵ���������ļ���" << PossamFile<<endl;
			cout << "Ĭ��ѵ���������ļ���" << NegsamFile << endl;
			//�����Ҫ�޸��ļ��������ļ���
			std::cout << "�Ƿ���Ҫ�޸�ѵ���ļ�����Y/N��";
			char tc=0;
			cin >> tc;
			if (tc == 'Y' || tc == 'y')
			{
				std::cout << "����ѵ���������ļ���";
				cin >> PossamFile;
				ifstream fin(PossamFile);
				while (!fin)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "����������ѵ���������ļ���";
					cin >> PossamFile;
					ifstream fin(PossamFile);
					if (fin)
						break;

				}
				fin.close();
				std::cout << "����ѵ���������ļ���";
				cin >> NegsamFile;
				ifstream fin2(NegsamFile);
				if (!fin2)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "����������ѵ���������ļ���";
					cin >> NegsamFile;
					ifstream fin2(NegsamFile);
					if (fin2)
						break;
				}
				fin2.close();

				tc = 0;
			}
			//ѵ���ļ�ȷ��֮��ȷ��ѵ���ļ�
			//���ز����ļ�

			double *TestData;
			double **tx1;
			double **tx2;
			double *ty1;
			double *ty2;
			//�����������
			char TestPosfile[200]="tp.dat";
			char TestNegfile[200]="tn.dat";
			//��һ����������������
			cout << "-------------------------------------" << endl;
			cout << "Ĭ�ϲ����������ļ���" << TestPosfile << endl;
			cout << "Ĭ�ϲ��Ը������ļ���" << TestNegfile << endl;
			std::cout << "�Ƿ���Ҫ�޸Ĳ�����������Y/N��";
			cin >> tc;
			if (tc == 'Y' || tc == 'y')
			{
				std::cout << "�������������:";	
				cin >> TestPosfile;
			
				ifstream fin(TestPosfile);
				while (!fin)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "��������������������ļ���";
					cin >> TestPosfile;
					ifstream fin(TestPosfile);
					if (fin)
						break;
				}
				fin.close();
				std::cout << "������Ը�����:";
				cin >> TestNegfile;
				ifstream fin2(TestNegfile);
				while (!fin2)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "������������Ը������ļ���";
					cin >> TestNegfile;
					ifstream fin2(TestNegfile);
					if (fin2)
						break;
				}
				fin2.close();
				tc = 0;
			}
			//ѵ����������
			cout << "��ʼ��������............." << endl;
			int PosNum = LineCount(PossamFile); //��������
			int NegNum = LineCount(NegsamFile); //��������
			SampleNum = PosNum + NegNum;  //��������


			//��ʼΪ��������ռ�
			x = new double*[SampleNum];
			if (x == NULL)
			{
				cout <<"�ڴ�ľ�" << endl;
				exit(1);
			}
			for (int i = 0; i < SampleNum; i++)
			{
				x[i] = new double[Diamension];
				if (x[i] == NULL)
				{
					cout << "�ڴ�ľ�" << endl;
					exit(1);
				}
			}
			y = new double[SampleNum];
			if (y == NULL)
			{
				cout << "�ڴ�ľ�" << endl;
				exit(1);
			}
			alpha = new double[SampleNum];
			if (alpha == NULL)
			{
				cout << "�ڴ�ľ�" << endl;
				exit(1);
			}
			dotCache = new double*[SampleNum];
			if (dotCache == NULL)
			{
				cout << "�ڴ�ľ�" << endl;
				exit(1);
			}
			for (int i = 0; i < SampleNum; i++)
			{
				dotCache[i] = new double[SampleNum];
				if (dotCache[i] == NULL)
				{
					cout << "�ڴ�ľ�" << endl;
					exit(1);
				}
			}
			kernel = new double*[SampleNum];
			if (kernel == NULL)
			{
				cout << "�ڴ�ľ�" << endl;
				exit(1);
			}
			for (int i = 0; i < SampleNum; i++)
			{
				kernel[i] = new double[SampleNum];
				if (kernel[i] == NULL)
				{
					cout << "�ڴ�ľ�" << endl;
					exit(1);
				}
			}
			errorCache = new double[SampleNum];
			if (errorCache == NULL)
			{
				cout << "�ڴ�ľ�" << endl;
				exit(1);
			}
			//�������
			b = 0;
			for (int i = 0; i < SampleNum; i++)
			{
				alpha[i] = 0;
				errorCache[i] = 0;
			}

			RamApply = 1;
			//������������
			int file1num = LineCount(TestPosfile); //ͳ�Ʋ�����������
			int file2num = LineCount(TestNegfile);
			int tn = file1num + file2num; //������������




			//
			//��ȡ�ļ�
			ifstream posfile(PossamFile);
			ifstream negfile(NegsamFile);

			//�����ڴ�
			DataCache = new double[SampleNum*Diamension];
			//����ѵ������
			for (int i = 0; i < SampleNum*(Diamension); i++)
			{
				if (i<PosNum*Diamension)
					posfile >> DataCache[i];
				else
					negfile >> DataCache[i];
			}
			//��������
			for (int i = 0; i < SampleNum; i++)
			{
				for (int j = 0; j < Diamension; j++)
				{
					x[i][j] = DataCache[j + (Diamension)*(i)];
				}
				if (i < PosNum)
					y[i] = 1;
				else
					y[i] = -1;
			}
			//�ͷ��ڴ�
			delete[] DataCache;

			posfile.close();
			negfile.close();			
			//ѵ�����ݳ�ʼ��
			dataInit(SampleNum, Diamension, gamma);
		
			//�������ݼ���
			ifstream file1(TestPosfile);
			//�����ڴ�
			TestData = new double[file1num*Diamension];
			tx1 = new double*[file1num];
			for (int i = 0; i < file1num; i++)
			{
				tx1[i] = new double[Diamension];
			}
			//tx2 = new double*[file2num*Diamension];
			ty1 = new double[file1num];
			//tx1 = new double[file1num*(Diamension + 1)];
			for (int i = 0; i < file1num*(Diamension); i++)
			{
				file1 >> TestData[i];
			}
			for (int i = 0; i < file1num; i++)
			{
				for (int j = 0; j < Diamension; j++)
				{
					tx1[i][j] = TestData[j + Diamension*i];
				}
				ty1[i] = 1;
			}
			//
			delete TestData;
			ifstream file2(TestNegfile);
			//�����ڴ�
			TestData = new double[file2num*Diamension];
			tx2 = new double*[file2num];
			for (int i = 0; i < file2num; i++)
			{
				tx2[i] = new double[Diamension];
			}
			//tx2 = new double*[file2num*Diamension];
			ty2 = new double[file2num];
			//tx1 = new double[file1num*(Diamension + 1)];
			for (int i = 0; i < file2num*(Diamension); i++)
			{
				file2 >> TestData[i];
			}
			for (int i = 0; i < file2num; i++)
			{
				for (int j = 0; j < Diamension; j++)
				{
					tx2[i][j] = TestData[j + Diamension*i];
				}
				ty2[i] = -1;
			}
			//
			delete TestData;

			//
			cout << "���ݴ������" << endl;
			int AutoUserC = 200;
			int AutoCount = 0;
			double ProbUser = 0.90;
			double AutoProb = 0;

			//��ѵ������
			cout << "Ĭ��ѵ������"<<AutoUserC<<" ����׼ȷ��="<<ProbUser<< endl;
			cout << "�Ƿ���Ҫ�޸�?(Y/N)";
			cin >> tc;
			if (tc == 'Y' || tc == 'y')
			{
				cout << "������ѵ������:";
				cin >> AutoUserC;
				cout << "������ѵ��׼ȷ��:";
				cin >> ProbUser;
			}
			//��ʼѵ��
			cout << "�Զ�ѵ����ʼ>>>>>>"<< endl;
			cout << ">>>>>>------------------->" << endl;
			//��¼��ʼʱ��
			long begintime = GetTickCount();
			//�ļ���¼
			ofstream log;
			log.open("AutoLog.txt", ios::app | ios::out);
			log << "-----------------------------------------------------------------------"<< endl;;
			log << "C=" << C << " Sigma=" << sigma << "Tolerence" << tolerance << endl;;
			log << "ѵ����������"<<PossamFile<<" ѵ����������"<<NegsamFile<< endl;
			log << "������������" << TestPosfile << " ���Ը�������" << TestNegfile << endl;
			//ѵ����֮����Ҫ���ԣ���׼ȷ�ʴﵽAutoProb x%֮���Զ�ѵ������
			//Ϊ�˱�����ִﲻ��ѵ����׼���趨һ���Զ�ѵ���Ĵ���AutoCount
			//**************************************---------ѭ��ѵ������--------------------------------------------************************//
			while (AutoCount < AutoUserC&&AutoProb <= ProbUser)
			{
				AutoCount++;//�ۼ�
				//memset(alpha, 0, 2000);
				//memset(errorCache, 0, 2000);//
				cout << "��"<<AutoCount<<"��ѵ��" << endl;
				log << "-----------------------------------------------------------------------" << endl;;
				log << "��" << AutoCount << "��ѵ��" <<endl;;
			
				//ÿһ�ζ�Ҫ���
				//���b
				//b = 0;
				iterCount = 0; //��������
				numChanged = 0;
				examineAll = true;
		
				while ((iterCount<maxIter) && (numChanged>0 || examineAll))
				{
					numChanged = 0;
					if (examineAll)
					{
						for (int i = 0; i < SampleNum; i++)
						{
							if (examineExample(i, SampleNum, C, gamma, tolerance))
								numChanged++;	
						}
					}
					else
					{
						for (int i = 0; i < SampleNum; i++)
						{
							if (alpha[i] != 0 && alpha[i] != C) 
							{
								if (examineExample(i, SampleNum, C, gamma, tolerance))
									numChanged++;
							}
						}
					}

					iterCount++;
					if (examineAll)
						examineAll = false;
					else if (numChanged == 0)
						examineAll = true;

				}



			int Nsv = 0;
			for (int i = 0; i < SampleNum; i++)
			{
				if (alpha[i]>0)
					Nsv++;       //ͳ��֧����������
				
			}
			log << "֧����������:" <<Nsv << endl;
			cout << "֧����������:" << Nsv << endl;
			for (int i = 0; i < SampleNum; i++)
			{
				//printf("alpha %d =  %f \n",i, alpha[i]);
				if (alpha[i] != 0)
				{	
					log << i << " Label=" << y[i] << ";" << " alpha" << i << " : " << alpha[i] << endl;//<<" " <<x[i][0] <<" "<<x[i][1]
					cout << i << " Label=" << y[i] << ";" << " alpha" << i << " : " << alpha[i] << endl;
				}
			}
			cout << "��" << AutoCount << "��ѵ������"<<endl<<"��ʼ����"<< endl;
			double probablity = 0.0;
			int correctCount = 0;	

			//���Բ���
			//----------------�����ļ�1----------------------------//
			for (int i = 0; i < file1num; i++)
			{
				double sum = 0.0;
				for (int j = 0; j < SampleNum; j++)
				{
					sum += alpha[j] * y[j] * kFunction(x[j], tx1[i], gamma, Diamension); //��֤ ���ߺ���
				}
				sum += b;
				if (sum>0)//&& ty1[i]>0
					correctCount++;
			}
			//

			//----------------�����ļ�2----------------------------//
			

			for (int i = 0; i < file2num; i++)
			{
				double sum = 0.0;
				for (int j = 0; j < SampleNum; j++)
				{
					sum += alpha[j] * y[j] * kFunction(x[j], tx2[i], gamma, Diamension); //��֤ ���ߺ���
				}
				sum += b;
				if (sum < 0)//&&ty2[i] < 0
					correctCount++;
			}
			

			//ÿ�β��Զ���¼

			//------------------------------------------//
			probablity = (double)correctCount / tn;
			log << "-----------Test -----------------------" << endl;
			log << "correctCount=" << correctCount << "; Probablity=" << probablity*100.0 << "%" << endl;
			
			
			std::cout << "correctCount=" << correctCount << "; Probablity=" << probablity*100.0 << "%" << endl;
			std::cout << "-------------------------------------------------------------------------" << endl;
			//

			//׼ȷ�ʼ�¼
			AutoProb = probablity;
			//-----------------------------------------------------------------------------//
			}
			//ѵ��ĩ
			//��ʱ����
			long endtime = GetTickCount();
			long diftime = endtime - begintime;
			log <<  "ѵ���ܼ�ʱ�� :" << (double)diftime / 1000 << "��" << endl;			
		    log.close();
			//�ڴ��ͷ�
			delete[]ty1;
			for (int i = 0; i < file1num; i++)
			{
				delete[] tx1[i];
			}
			delete[] tx1;
			tx1 = NULL;
			file1.close();
			delete[] ty2;
			for (int i = 0; i < file2num; i++)
			{
				delete[] tx2[i];
			}
			delete[] tx2;
			tx2 = NULL;
			file2.close();

			std::cout << "ѵ���ܼ�ʱ�� :" << (double)diftime / 1000 << "��" << endl;
			std::cout << "ѵ����ɣ�"<<endl;
			std::cout << "��������:";
		}
		else
		{
			//memset(choice, 0, 5);
			//cout << "��������:";
			//char *choice;
			cin >> choice;
		}

	}

	

	return 0;
}

int LineCount(char *filename)
{
	ifstream file;
	int LineNum = 0;
	string tmp;
	file.open(filename, ios::in);
	if (file.fail())
		return 0;
	else
	{
		while (getline(file,tmp))
		{
			LineNum++;
		}
		
	}
	file.close();
	return LineNum;
}

