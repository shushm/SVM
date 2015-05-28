// SVM.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

//#define HOG
#define TEST

#ifdef HOG
#define	  n  3780			//定义维数
#define   m  200			//定义样本个数
int dataChoice = 1;			//样本选项 0---测试用样本 维数3 数量 20 ；1---hog样本 维数：3780 数量200
#endif // HOG

#ifdef TEST
#define	  n  2				//定义维数
#define   m  20 			//定义样本个数
int dataChoice = 0;			//样本选项 0---测试用样本 维数3 数量 20 ；1---hog样本 维数：3780 数量200
#endif // TEST


double x[m][n] = { 0 };		//样本特征
int y[m];					//样本标签

double b = 0.0;
double alpha[m] = { 0 };	//拉格朗日乘子
double C = 0.05;			//惩罚因子
double tolerance = 0.001;	//松弛变量

double errorCache[m];		//误差缓存
double eps = 0.001;			//终止条件的差值 

//
// rbf kernel for exp(-gamma*|u-v|^2) 高斯径向基核函数（无变形）
//
double sigma =10; //
double gamma = 1/(sigma*sigma);

//
//	训练用参数
//
int maxIter =8000; //定义最大迭代次数
int iterCount = 0; //迭代计数
int numChanged = 0;
bool examineAll = true;

//点积缓存
double dotCache[m][m];
//核缓存
double kernel[m][m];



double dot(double x[], int xLen, double y[],int yLen)
{
	double sum=0;
	int i=0, j = 0;
	while (i < xLen&&j < yLen)
	{
		sum+=x[i] * y[j];   //样本特征维数固定 若不固定则需要比较长度 
		i++;
		j++;
	}
	return sum;
}

//
//训练时用的核函数
//
double kernelFunction(int i1, int i2){
	double result = 0.0;
	result = exp(-gamma * (dotCache[i1][i1] + dotCache[i2][i2] - 2 * dotCache[i1][i2])); 
	return result;
}
//
//
//
double dataInit(int M,int N) //样本个数：M    维数：N
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
		for (int j = 0; j < M; j++)
		{
			kernel[i][j] = kernelFunction(i, j);
		}
	}
	return 1;
}

double learnFunc(int k)  //学习函数U ,算误差
{
	double sum = 0;
	for (int i = 0; i < m; i++)
	{
		sum += alpha[i] * y[i] * kernel[i][k];
	}
	sum = sum + b;
	return sum;
}
double calError(int k)
{
	double error = learnFunc(k) - y[k];
	return error;

}

double updateErrorCache(int k)
{
	double error = calError(k);
	errorCache[k] = error;

	return 1;
}

int selectMaxJ(double E1)
{
	int i2 = -1;
	double tmax = 0.0;
	for (int k = 0; k < m;k++)
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
//产生随机数,随机选择i1，但不能等于i2
//
int randomSelect(int i1)
{
	int i2 = 0;
	do
	{
		srand((int)time(0));
		i2 = rand() % m;  //
	} while (i2 == i1);

	return i2;
}
/**
*  单个迭代
*  input i1,i2
*  return 0/1
*/ 
int takeStep(int i1, int i2)
{

	if (i1 == i2) return 0;
	double alpha1 = alpha[i1];
	double alpha2 = alpha[i2];
	int y1 = y[i1];
	int y2 = y[i2];
	double E1 = 0;
	double E2 = 0;
	double a1, a2;
	int s = y1*y2;
	//init L,H
	double L, H;
	//compute E1
	if (0 < alpha1 && alpha1 < C)
		E1 = errorCache[i1];
	else
		E1 = calError(i1);
	//compute E2
	if (0 < alpha2 && alpha2 < C)
		E2 = errorCache[i2];
	else
		E2 = calError(i2);
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
	//更新误差缓存
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

	updateErrorCache(i1);
	updateErrorCache(i2);

	alpha[i1] = a1;
	alpha[i2] = a2;

	return 1;
}
//
//预测用核函数  建议 gamma=0.5
//
double kFunction(double x[],double y[],double gamma)
{
	double sum = 0;
	int i = 0, j = 0;
	while (i<n||j<n)
	{
		if (i== j)
		{
			double d = x[i] - y[j];
			sum += d*d;
			i++;
			j++;
		}
		else if (i>j)
		{
			sum += y[j]*y[j];
			j++;
		}
		else
		{
			sum += x[i] * x[i];
			i++;
		}
	}
	return exp(-gamma*sum);   //高斯核
}

double examineExample(int i1)
{
	double y1 = y[i1];
	double alpha1 = alpha[i1];
	double E1 = 0;

	if (0 < alpha1&&alpha1 < C)
		E1 = errorCache[i1];
	else
		E1 = calError(i1);

	//printf(" E1  =  %f\n",E1);
	double r1 = E1*y1;  
	//printf(" r1  =  %f\n", r1);
	if (((r1<-tolerance) && alpha1<C) || (r1>tolerance&&alpha1>0))
	{
		//选择E1-E2差最大的点
		int i2 = selectMaxJ(E1);
		if (i2 >= 0)
		{
			if  (takeStep(i1, i2))
				return 1;
		}
		//选择0<alpha<C的点随机选取一个起始点
		int k0 = randomSelect(i1);
		//printf(" k0  =  %d \n", k0);
		for (int k = k0; k < m+k0; k++)
		{
			i2 = k%m;
			//printf(" i1  =  %d \n", i1);
			if (0<alpha[i2]&&alpha[i2]<C)
			{
				if (takeStep(i1, i2))
					return 1;
			}

		}

		//如果不符合，再遍历全部点
		k0 = randomSelect(i1);
		//printf(" k0  =  %d \n", k0);
		for (int k = k0; k < m+k0; k++)
		{
			i2 = k%m;
			if (takeStep(i1, i2))
				return 1;
			

		}

	}

	return 0;
}


double predict()
{
	double probablity = 0.0;
	int correctCount = 0;
	for (int i = 0; i < m; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < m; j++)
		{
			sum += alpha[j] * y[j] * kFunction(x[j], x[i], gamma); //验证 决策函数
		}
		sum += b;
		if ((sum>0 && y[i]>0) || (sum < 0 && y[i] < 0))
			correctCount++;
	}
	probablity = (double) correctCount / m;
	printf("correctCount= %d , probablity= %f \n", correctCount, probablity);
	return probablity;

}

int _tmain(int argc, _TCHAR* argv[])
{
	double * DataCache;
	printf("Train Begin ...\n");
	printf("核函数选取-高斯核 \n");
	printf("sigma= %f \n",gamma);
	printf("Loading Data \n");
	//打开文件
	//



	//数据加载到DataCche
	/*for (int i = 0; i < m*n; i++)
	{
		if (i<m*n / 2)
		negfile >> DataCache[i];
		else
			negfile >> DataCache[i];
	}*/
	//for (int i = 0; i < m*n; i++)
	//{
	//	cout << "DataCache["<<i<<"]="<<DataCache[i]<<"\n";
	//}
	
	printf("样本个数= %d \n", m);
	printf("样本维数= %d \n", n);
	
	if (dataChoice==0)
	{
		ifstream infile("TrainData.txt");
		DataCache = new double[m*(n+1)];
		for (int i = 0; i < m*(n + 1); i++)
		{
			infile >> DataCache[i];
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				x[i][j] = DataCache[j+(n+1)*(i)];
			}
			y[i] = DataCache[(n+1)*(i + 1)-1];
		}
		//释放内存
		delete DataCache;
	}
	else if (dataChoice == 1)  //hog样本
	{
		ifstream posfile("positive.dat");
		ifstream negfile("negative.dat");
		//double DataCache[m*(n )];
		//分配内存
		DataCache = new double[m*n];
		//加载数据
		for (int i = 0; i < m*(n ); i++)
		{
			if (i<m*n/2)
				posfile>> DataCache[i];
			else
				negfile >> DataCache[i];
		}
		//归类样本
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				x[i][j] = DataCache[j + (n)*(i)];
			}
			if (i < 100)   //100正样本
				y[i] = 1;
			else if (i >= 100) //100负样本
				y[i] = -1;
		}
		//释放内存
		delete DataCache;
		
	}
	
	

	printf("Load Success\nBegin to Intialize data\n");

	dataInit(m,n);
	printf("Intialize OK \n");
	while ((iterCount<maxIter) && (numChanged>0 || examineAll))
	{
		numChanged = 0;
		if (examineAll)
		{
			for (int i = 0; i < m; i++)
			{
				if (examineExample(i))
					numChanged++;
				printf("examineAll=%d \n", examineAll);
				printf("numChanged=%d \n", numChanged);
			}
		}
		else
		{
			for (int i = 0; i < m; i++)
			{
				if (alpha[i] != 0 && alpha[i] != C)
				{
					if (examineExample(i))
						numChanged++;
					printf("examineAll=%d\n", examineAll);
					printf("numChanged=%d \n", numChanged);
				}
			}
		}
		//printf("numChanged=%d \n", numChanged);
		iterCount++;
		printf("iterCount =  %d \n", iterCount);
		if (examineAll)
			examineAll = false;
		else if (numChanged == 0)
			examineAll = true;			

	}
	printf("Train Success\n");
		for (int i = 0; i < m; i++)
		{
			printf("alpha %d =  %f \n",i, alpha[i]);
		}
		printf("b =  %f \n", b);

		predict();
	return 0;
}