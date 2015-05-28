// SVM.cpp : 定义控制台应用程序的入口点。
//-------------------------------------------------------//
//程序名：支持向量机RBF核训练测试程序
//作者：沈华明
//最新版本：1.1 alpha 更改说明 修改1.0beta版本的样本数量限制
//最后修改时间：2015-5-28||13：07
//之前版本:
//--------------------------------------------------------//
//1.0 beta 更改说明 增加自动验证并训练，达到预期准确率结果
//1.0 alpha         修正测试不准确问题
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
double x[2000][5000] = { 0 };		//样本特征
double y[2000];					//样本标签

double b = 0.0;
double alpha[2000] = { 0 };	//拉格朗日乘子


double dotCache[2000][2000];
//核缓存
double kernel[2000][2000];
double errorCache[2000];		//误差缓存
double eps = 0.001;			//终止条件的差值 
*/
//
// rbf kernel for exp(-gamma*|u-v|^2) 高斯径向基核函数（无变形）
//


//
//	训练用参数
//

/*
//点积缓存
double dotCache[m][m];
//核缓存
double kernel[m][m];


//
double x[m][n] = { 0 };		//样本特征
double y[m];					//样本标签

double b = 0.0;
double alpha[m] = { 0 };	//拉格朗日乘子
double C = 5;			//惩罚因子
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
int maxIter =5000; //定义最大迭代次数
int iterCount = 0; //迭代计数
int numChanged = 0;
bool examineAll = true;

//点积缓存
double dotCache[m][m];
//核缓存
double kernel[m][m];

*/

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
double kernelFunction(int i1, int i2,double gamma)
{
	double result = 0.0;
	result = exp(-gamma * (dotCache[i1][i1] + dotCache[i2][i2] - 2 * dotCache[i1][i2])); 
	return result;
}
//
//
//
double dataInit(int M,int N,double gamma) //样本个数：M    维数：N
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
			cout << "内存耗尽" << endl;
			exit(1);
		}
		for (int j = 0; j < M; j++)
		{
			kernel[i][j] = kernelFunction(i, j,gamma);
		}
	}
	return 1;
}

double learnFunc(int k,int SampleNum)  //学习函数U ,算误差
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
//产生随机数,随机选择i1，但不能等于i2
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
*  单个迭代
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

	updateErrorCache(i1, SampleNum);
	updateErrorCache(i2, SampleNum);

	alpha[i1] = a1;
	alpha[i2] = a2;

	return 1;
}
//
//预测用核函数  建议 gamma=0.5
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
	return exp(-gamma*sum);   //高斯核
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
		//选择E1-E2差最大的点
		int i2 = selectMaxJ(E1, SampleNum,C);
		if (i2 >= 0)
		{
			if (takeStep(i1, i2, SampleNum,C))
				return 1;
		}
		//选择0<alpha<C的点随机选取一个起始点
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

		//如果不符合，再遍历全部点
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

//testfile1 正测试样本 
//testfile2 负测试样本

double predict(char *testfile1,char *testfile2, int  SampleNum, int Diamension,double gamma)
{
	double probablity = 0.0;
	int correctCount = 0;
	
	int file1num = 0;
	int file2num = 0;
	file1num = LineCount(testfile1); //统计测试样本数量
	file2num = LineCount(testfile2);
	int tn = file1num+file2num; //测试样本数量
	std::cout << "-------------------------------------------------" << endl;
	std::cout << "测试样本数量：" << tn << endl;
	//打开测试样本1
	
	double *TestData;
	double **tx1;
	double **tx2;
	double *ty1;
	double *ty2;
	//----------------测试文件1----------------------------//
	ifstream file1(testfile1);
	//申请内存
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
			sum += alpha[j] * y[j] * kFunction(x[j], tx1[i], gamma, Diamension); //验证 决策函数
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
	//----------------测试文件2----------------------------//
	ifstream file2(testfile2);
	//申请内存
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
			sum += alpha[j] * y[j] * kFunction(x[j], tx2[i], gamma, Diamension); //验证 决策函数
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
	//训练相关参数
	double sigma = 5; //
	double gamma = 1 / (sigma*sigma);
	double C = 4;			//惩罚因子
	double tolerance = 0.001;	//松弛变量
	//
	int SampleNum = 0;  //总样本数
	int Diamension =3780;  //样本维数 默认3780hog特征维数
	//
	int maxIter = 50000; //定义最大迭代次数
	int iterCount = 0; //迭代计数
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
	std::cout << "---------------------输入选项---------------------------" << endl;
	std::cout << "-               help          ----帮助选项             -" << endl;
	std::cout << "-               Train         ----训练选项             -" << endl;
	std::cout << "-               Test          ----测试选项             -" << endl;
	std::cout << "-               Auto          ----自动训练选项         -" << endl;
	std::cout << "-               quit          ----退出程序             -" << endl;
	std::cout << "--------------------------------------------------------" << endl;
	std::cout << "-                   使用说明                           -" << endl;
	std::cout << "-               第一步      训练                       -" << endl;
	std::cout << "-               第二步--测试或者训练                   -" << endl;
	std::cout << "-               训练模型追加在model文件                -" << endl;
	std::cout << "-               测试记录追加在log文件内                -" << endl;
	std::cout << "--------------------------------------------------------" << endl;
	std::cout << "Enjoy yourself!" <<endl<< endl;
	std::cout << "输入命令:";
	 char choice[5];
	cin >> choice;
	//训练样本名
	char PossamFile[100]="p1.dat";
	char NegsamFile[100]="n1.dat";
	//
	//FirstWrite = 1;
	//训练用的数据 内存是否申请标志，一旦申请过了，再次申请前需要释放内存
	char RamApply = 0;
	//
	while (1)
	{
		if ((_strnicmp(choice,"help",4))==0)
		{
			memset(choice, 0, 5);
			std::cout << "<<------------------输入选项------------------->>" << endl;
			std::cout << "               help          ----帮助选项        " << endl;
			std::cout << "               Train         ----训练选项        " << endl;
			std::cout << "               Test          ----测试选项        " << endl;
			std::cout << "               Auto          ----自动训练选项    " << endl;
			std::cout << "               quit          ----退出程序        " << endl;
			std::cout << "-------------------------------------------------" << endl;

			std::cout << "输入命令:";
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
			//数据清空
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
			//清空b
			b = 0;
			iterCount = 0; //迭代计数
			numChanged = 0;
			examineAll = true;
			//
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "核函数 RBF--> exp((-||x-x'||^2)/(sigma^2))" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "默认的参数为>>>>>" << endl;
			std::cout << "C=" << C << "; sigma=" << sigma << endl;
			std::cout << "最大循环次数=" << maxIter << "; 松弛变量=" << tolerance << endl;
			std::cout << "终止条件精度=" << eps << endl;
			//
			char ic;
			std::cout << "是否要更改参数？（Y/N）";
			cin >> ic;
			if (ic == 'Y' || ic == 'y')
			{
				std::cout << "请输入惩罚因子C=";
				cin >> C;
				std::cout << "请输入参数sigma=";
				std::cin >> sigma;
				std::cout << "松弛变量=";
				cin >> tolerance;
				std::cout << "最大循环次数=";
				cin >> maxIter;
				std::cout << "终止条件精度=";
				cin >> eps;
			}
			std::cout << "---------------------------------------------" << endl;

			//样本数量

			//样本维数 hog:3780
			cout << "维数=" << Diamension << endl;
			char dc;
			cout << "是否需要改变样本维数？(Y/N)";
			cin >> dc;
			if (dc == 'Y' || dc == 'y')
			{
				std::cout << "请输入样本维数：";
				cin >> Diamension;
			}

			//cout <<"样本维数为：" <<Diamension << endl;
			cout <<"维数="<<Diamension << endl;
			std::cout << "输入训练正样本文件：";
			cin >> PossamFile;
			ifstream fin(PossamFile);
			while (!fin)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "请重新输入训练正样本文件：";
				cin >> PossamFile;
				ifstream fin(PossamFile);
				if (fin)
					break;
				//break;
				//return 0;
			//	std::cout << "输入训练正样本文件：";
			//	cin >> PossamFile;

			}
			fin.close();
			std::cout << "输入训练负样本文件：";
			cin >> NegsamFile;
			ifstream fin2(NegsamFile);
			if (!fin2)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "请重新输入训练负样本文件：";
				cin >> NegsamFile;
				ifstream fin2(NegsamFile);
				if (fin2)
					break;
			}
			fin2.close();
			std::cout << "样本文件正常" << endl;
			std::cout << "准备开始>>>>>等待可能时间稍长>>>>>>>>>" << endl;


			//记录开始时间
			long begintime = GetTickCount();


			//cout << "现在时间："<<ctime(&timeval) << endl;

			int PosNum = LineCount(PossamFile); //正样本数
			int NegNum = LineCount(NegsamFile); //负样本数
			SampleNum = PosNum + NegNum;  //总样本数
			std::cout << "样本个数= " << SampleNum << endl;
			std::cout << "样本维数= " << Diamension << endl;
			//开始为样本申请空间
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
					cout << "内存耗尽"<< endl;
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
			
			//标记ram已经申请到
			RamApply = 1;

			//
			ofstream ofile;
			//日志文件

			//文件名 目前用户不可修改
			ofile.open("model.txt",ios::out|ios::app); //输出模型文件
			//

			//log.se;
			//模型数据对文件输出
			ofile << "----------------------------------------------------" << endl;
			ofile << "----------------The SVM model:---------------------- " << endl;
			ofile << "----------------------------------------------------" << endl;
			//屏幕显示开始训练
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "Train Begin ..." << endl;
			std::cout << "---->>>>>>>>>>>>>>>" << endl;
			std::cout << "核函数选取-高斯核" << endl;
			std::cout << "C=" << C << "; sigma=" << sigma << endl;
			std::cout << "最大循环次数=" << maxIter << "; 松弛变量=" << tolerance << endl;
			std::cout << "终止条件精度=" << eps << endl;
			//
			ofile << "Kernel: RBF" << endl;
			//
			std::cout << "Loading Data---->>>>>>>>>>>>>" << endl;
			//打开文件
			//
			ofile << "gamma = " << gamma << endl;
			ofile << "C = " << C << endl;
			ofile << "训练样本=" << PossamFile << ";" << NegsamFile << endl;

			//读取文件
			ifstream posfile(PossamFile);
			ifstream negfile(NegsamFile);

			//分配内存
			DataCache = new double[SampleNum*Diamension];
			//加载数据
			for (int i = 0; i < SampleNum*(Diamension); i++)
			{
				if (i<PosNum*Diamension)
					posfile >> DataCache[i];
				else
					negfile >> DataCache[i];
			}
			//归类样本
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
			//释放内存
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
			std::cout << "训练总计循环次数 :" << iterCount << "次" << endl;
			long endtime = GetTickCount();
			//cout << "现在时间：" << ctime(&timeval2) << endl;
			long diftime = endtime - begintime;
			std::cout << "训练总计时间 :" << (double)diftime / 1000 << "秒" << endl;

			int Nsv = 0;
			for (int i = 0; i < SampleNum; i++)
			{
				if (alpha[i]>0)
					Nsv++;       //统计支持向量个数
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
			std::cout << "乘子结果在模型文件内" << endl;
			std::printf("b =  %f \n", b);
			ofile << "b= " << b << endl;
			ofile.close();
			std::cout << "-----------------------------------------------------" << endl;
			std::cout << "输入命令:";
		}
		else if ((_strnicmp(choice, "test", 4)) == 0)
		{
			memset(choice, 0, 5);
			//测试数据记录
			log.open("Testlog.txt",ios::app|ios::out);
			log << "-----------Test Start-----------------------" << endl;
			log << "C=" << C << " ;sigma=" << sigma << " ;循环次数 =" << maxIter << endl;
			log << "eps=" << eps << ";tolerence=" << tolerance << endl;
			log << "训练样本=" << PossamFile << ";" << NegsamFile << endl;
			//测试显示数据
			std::cout << ">>>>开始测试-------->>>>>>>>>" << endl;
			std::cout << "----------------------------------" << endl;
			std::cout << "校对训练数据---->>>>>"<<endl;
			std::cout << "C=" << C << " ;sigma=" << sigma << " ;循环次数 =" << maxIter << endl;
			std::cout << "eps=" << eps << ";tolerence=" << tolerance << endl;
			std::cout << "训练样本=" << PossamFile << ";" << NegsamFile << endl;
			std::cout << "----------------------------------" << endl;
			
			std::cout << "输入测试正样本:";
			char TestPosfile[200];
			cin >> TestPosfile;

			ifstream fin(TestPosfile);
			while (!fin)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "请重新输入测试正样本文件：";
				cin >> TestPosfile;
				ifstream fin(TestPosfile);
				if (fin)
					break;
			}
			fin.close();


			char TestNegfile[200];
			std::cout << "输入测试负样本:";
			cin >> TestNegfile;

			ifstream fin2(TestNegfile);
			while (!fin2)
			{
				std::cout << "Error Can not find the file!" << endl;
				std::cout << "请重新输入测试负样本文件：";
				cin >> TestNegfile;
				ifstream fin2(TestNegfile);
				if (fin2)
					break;
			}
			fin2.close();

			log << "测试样本文件=" << TestPosfile << ";" <<TestNegfile << endl;
			std::cout << "如果文件较大，花费时间较长，请耐心等待>>>>>>>>>" << endl;
			predict(TestPosfile, TestNegfile, SampleNum, Diamension, gamma);
			std::cout << "-----------------------------------------------------" << endl;
			std::cout << "输入命令:";
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
			std::cout << ">>>>>>>>谢谢使用>>>>>>>>>>>>>" << endl;
			//Sleep(3 * 1000);
			return 0;
		}//-------------------------------自动训练部分-------------------------------//
		else if ((_strnicmp(choice, "auto", 4)) == 0)//--------------------------------------------------------//
		{
			memset(choice, 0, 5);
			std::cout << "已选择自动训练>>>>>>>>>>>>>>>" << endl;
			//已经申请过X Y的内存 需要释放再申请
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
			//每一次进入都需要数据清空
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
			//清空b
			b = 0;
			iterCount = 0; //迭代计数
			numChanged = 0;
			examineAll = true;
			//
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "核函数 RBF--> exp((-||x-x'||^2)/(sigma^2))" << endl;
			std::cout << "-------------------------------------------------" << endl;
			std::cout << "默认的参数为>>>>>" << endl;
			std::cout << "C=" << C << "; sigma=" << sigma << endl;
			std::cout << "最大循环次数=" << maxIter << "; 松弛变量=" << tolerance << endl;
			std::cout << "终止条件精度=" << eps << endl;
			//
			char ic;
			std::cout << "是否要更改训练参数？（Y/N）";
			cin >> ic;
			if (ic == 'Y' || ic == 'y')
			{
				std::cout << "请输入惩罚因子C=";
				cin >> C;
				std::cout << "请输入参数sigma=";
				std::cin >> sigma;
				std::cout << "松弛变量=";
				cin >> tolerance;
				std::cout << "最大循环次数=";
				cin >> maxIter;
				std::cout << "终止条件精度=";
				cin >> eps;
			}
			std::cout << "---------------------------------------------" << endl;

			//样本维数 hog:3780
			cout << "维数=" << Diamension << endl;
			char dc=0;
			cout << "是否需要改变样本维数？(Y/N)";
			cin >> dc;
			if (dc == 'Y' || dc == 'y')
			{
				std::cout << "请输入样本维数：";
				cin >> Diamension;
				cout << "维数=" << Diamension << endl;
				dc = 0;
			}

			
			cout << "-------------------------------------"<< endl;
			cout << "默认训练正样本文件：" << PossamFile<<endl;
			cout << "默认训练负样本文件：" << NegsamFile << endl;
			//如果需要修改文件则输入文件名
			std::cout << "是否需要修改训练文件？（Y/N）";
			char tc=0;
			cin >> tc;
			if (tc == 'Y' || tc == 'y')
			{
				std::cout << "输入训练正样本文件：";
				cin >> PossamFile;
				ifstream fin(PossamFile);
				while (!fin)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "请重新输入训练正样本文件：";
					cin >> PossamFile;
					ifstream fin(PossamFile);
					if (fin)
						break;

				}
				fin.close();
				std::cout << "输入训练负样本文件：";
				cin >> NegsamFile;
				ifstream fin2(NegsamFile);
				if (!fin2)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "请重新输入训练负样本文件：";
					cin >> NegsamFile;
					ifstream fin2(NegsamFile);
					if (fin2)
						break;
				}
				fin2.close();

				tc = 0;
			}
			//训练文件确认之后确认训练文件
			//加载测试文件

			double *TestData;
			double **tx1;
			double **tx2;
			double *ty1;
			double *ty2;
			//输入测试样本
			char TestPosfile[200]="tp.dat";
			char TestNegfile[200]="tn.dat";
			//第一步输入正测试样本
			cout << "-------------------------------------" << endl;
			cout << "默认测试正样本文件：" << TestPosfile << endl;
			cout << "默认测试负样本文件：" << TestNegfile << endl;
			std::cout << "是否需要修改测试样本：（Y/N）";
			cin >> tc;
			if (tc == 'Y' || tc == 'y')
			{
				std::cout << "输入测试正样本:";	
				cin >> TestPosfile;
			
				ifstream fin(TestPosfile);
				while (!fin)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "请重新输入测试正样本文件：";
					cin >> TestPosfile;
					ifstream fin(TestPosfile);
					if (fin)
						break;
				}
				fin.close();
				std::cout << "输入测试负样本:";
				cin >> TestNegfile;
				ifstream fin2(TestNegfile);
				while (!fin2)
				{
					std::cout << "Error Can not find the file!" << endl;
					std::cout << "请重新输入测试负样本文件：";
					cin >> TestNegfile;
					ifstream fin2(TestNegfile);
					if (fin2)
						break;
				}
				fin2.close();
				tc = 0;
			}
			//训练样本数量
			cout << "开始处理数据............." << endl;
			int PosNum = LineCount(PossamFile); //正样本数
			int NegNum = LineCount(NegsamFile); //负样本数
			SampleNum = PosNum + NegNum;  //总样本数


			//开始为样本申请空间
			x = new double*[SampleNum];
			if (x == NULL)
			{
				cout <<"内存耗尽" << endl;
				exit(1);
			}
			for (int i = 0; i < SampleNum; i++)
			{
				x[i] = new double[Diamension];
				if (x[i] == NULL)
				{
					cout << "内存耗尽" << endl;
					exit(1);
				}
			}
			y = new double[SampleNum];
			if (y == NULL)
			{
				cout << "内存耗尽" << endl;
				exit(1);
			}
			alpha = new double[SampleNum];
			if (alpha == NULL)
			{
				cout << "内存耗尽" << endl;
				exit(1);
			}
			dotCache = new double*[SampleNum];
			if (dotCache == NULL)
			{
				cout << "内存耗尽" << endl;
				exit(1);
			}
			for (int i = 0; i < SampleNum; i++)
			{
				dotCache[i] = new double[SampleNum];
				if (dotCache[i] == NULL)
				{
					cout << "内存耗尽" << endl;
					exit(1);
				}
			}
			kernel = new double*[SampleNum];
			if (kernel == NULL)
			{
				cout << "内存耗尽" << endl;
				exit(1);
			}
			for (int i = 0; i < SampleNum; i++)
			{
				kernel[i] = new double[SampleNum];
				if (kernel[i] == NULL)
				{
					cout << "内存耗尽" << endl;
					exit(1);
				}
			}
			errorCache = new double[SampleNum];
			if (errorCache == NULL)
			{
				cout << "内存耗尽" << endl;
				exit(1);
			}
			//数据清空
			b = 0;
			for (int i = 0; i < SampleNum; i++)
			{
				alpha[i] = 0;
				errorCache[i] = 0;
			}

			RamApply = 1;
			//测试样本数量
			int file1num = LineCount(TestPosfile); //统计测试样本数量
			int file2num = LineCount(TestNegfile);
			int tn = file1num + file2num; //测试样本数量




			//
			//读取文件
			ifstream posfile(PossamFile);
			ifstream negfile(NegsamFile);

			//分配内存
			DataCache = new double[SampleNum*Diamension];
			//加载训练数据
			for (int i = 0; i < SampleNum*(Diamension); i++)
			{
				if (i<PosNum*Diamension)
					posfile >> DataCache[i];
				else
					negfile >> DataCache[i];
			}
			//归类样本
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
			//释放内存
			delete[] DataCache;

			posfile.close();
			negfile.close();			
			//训练数据初始化
			dataInit(SampleNum, Diamension, gamma);
		
			//测试数据加载
			ifstream file1(TestPosfile);
			//申请内存
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
			//申请内存
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
			cout << "数据处理完成" << endl;
			int AutoUserC = 200;
			int AutoCount = 0;
			double ProbUser = 0.90;
			double AutoProb = 0;

			//总训练次数
			cout << "默认训练次数"<<AutoUserC<<" 期望准确率="<<ProbUser<< endl;
			cout << "是否需要修改?(Y/N)";
			cin >> tc;
			if (tc == 'Y' || tc == 'y')
			{
				cout << "请输入训练次数:";
				cin >> AutoUserC;
				cout << "请输入训练准确率:";
				cin >> ProbUser;
			}
			//开始训练
			cout << "自动训练开始>>>>>>"<< endl;
			cout << ">>>>>>------------------->" << endl;
			//记录开始时间
			long begintime = GetTickCount();
			//文件记录
			ofstream log;
			log.open("AutoLog.txt", ios::app | ios::out);
			log << "-----------------------------------------------------------------------"<< endl;;
			log << "C=" << C << " Sigma=" << sigma << "Tolerence" << tolerance << endl;;
			log << "训练正样本："<<PossamFile<<" 训练负样本："<<NegsamFile<< endl;
			log << "测试正样本：" << TestPosfile << " 测试负样本：" << TestNegfile << endl;
			//训练完之后需要测试，当准确率达到AutoProb x%之后自动训练结束
			//为了避免出现达不到训练标准，设定一个自动训练的次数AutoCount
			//**************************************---------循环训练部分--------------------------------------------************************//
			while (AutoCount < AutoUserC&&AutoProb <= ProbUser)
			{
				AutoCount++;//累计
				//memset(alpha, 0, 2000);
				//memset(errorCache, 0, 2000);//
				cout << "第"<<AutoCount<<"次训练" << endl;
				log << "-----------------------------------------------------------------------" << endl;;
				log << "第" << AutoCount << "次训练" <<endl;;
			
				//每一次都要清空
				//清空b
				//b = 0;
				iterCount = 0; //迭代计数
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
					Nsv++;       //统计支持向量个数
				
			}
			log << "支持向量个数:" <<Nsv << endl;
			cout << "支持向量个数:" << Nsv << endl;
			for (int i = 0; i < SampleNum; i++)
			{
				//printf("alpha %d =  %f \n",i, alpha[i]);
				if (alpha[i] != 0)
				{	
					log << i << " Label=" << y[i] << ";" << " alpha" << i << " : " << alpha[i] << endl;//<<" " <<x[i][0] <<" "<<x[i][1]
					cout << i << " Label=" << y[i] << ";" << " alpha" << i << " : " << alpha[i] << endl;
				}
			}
			cout << "第" << AutoCount << "次训练结束"<<endl<<"开始测试"<< endl;
			double probablity = 0.0;
			int correctCount = 0;	

			//测试部分
			//----------------测试文件1----------------------------//
			for (int i = 0; i < file1num; i++)
			{
				double sum = 0.0;
				for (int j = 0; j < SampleNum; j++)
				{
					sum += alpha[j] * y[j] * kFunction(x[j], tx1[i], gamma, Diamension); //验证 决策函数
				}
				sum += b;
				if (sum>0)//&& ty1[i]>0
					correctCount++;
			}
			//

			//----------------测试文件2----------------------------//
			

			for (int i = 0; i < file2num; i++)
			{
				double sum = 0.0;
				for (int j = 0; j < SampleNum; j++)
				{
					sum += alpha[j] * y[j] * kFunction(x[j], tx2[i], gamma, Diamension); //验证 决策函数
				}
				sum += b;
				if (sum < 0)//&&ty2[i] < 0
					correctCount++;
			}
			

			//每次测试都记录

			//------------------------------------------//
			probablity = (double)correctCount / tn;
			log << "-----------Test -----------------------" << endl;
			log << "correctCount=" << correctCount << "; Probablity=" << probablity*100.0 << "%" << endl;
			
			
			std::cout << "correctCount=" << correctCount << "; Probablity=" << probablity*100.0 << "%" << endl;
			std::cout << "-------------------------------------------------------------------------" << endl;
			//

			//准确率记录
			AutoProb = probablity;
			//-----------------------------------------------------------------------------//
			}
			//训练末
			//计时结束
			long endtime = GetTickCount();
			long diftime = endtime - begintime;
			log <<  "训练总计时间 :" << (double)diftime / 1000 << "秒" << endl;			
		    log.close();
			//内存释放
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

			std::cout << "训练总计时间 :" << (double)diftime / 1000 << "秒" << endl;
			std::cout << "训练完成！"<<endl;
			std::cout << "输入命令:";
		}
		else
		{
			//memset(choice, 0, 5);
			//cout << "输入命令:";
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

