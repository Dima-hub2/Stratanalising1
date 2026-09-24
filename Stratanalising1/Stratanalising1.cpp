#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <windows.h>
using namespace std;

int main()
{
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    srand((unsigned)time(0));

    int N, M, K;
    cout << "Количество объектов N: ";
    cin >> N;
    cout << "Количество признаков M: ";
    cin >> M;
    cout << "Количество классов K: ";
    cin >> K;
    // analcatdata_creditscore iris34_test analcatdata_authorship dermatology

    ifstream file("dermatology.txt");
    
    // a[i][0..M-1] — признаки, [i][M] — класс
    float** a = new float* [N];
    for (int i = 0; i < N; i++) {
        a[i] = new float[M + 1];
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            file >> a[i][j];
        }
        file >> a[i][M];
    }
    file.close();

    cout << endl << "Прочитано: " << N << " объектов, " << M << " признаков, " << K << " классов" << endl << endl;

    // Дележка 80/20
    int trainSize = (int)(N * 0.8);
    int testSize = N - trainSize;

    cout << "Тренировочные: " << trainSize << ", Тестовые: " << testSize << endl << endl;

    float** train = new float* [trainSize];    
    for (int i = 0; i < trainSize; i++) {
        train[i] = new float[M + 1];
    }

    float** test = new float* [testSize];
    for (int i = 0; i < testSize; i++) {
        test[i] = new float[M + 1];
    }

    bool* used = new bool[N];
    for (int i = 0; i < N; i++) {
        used[i] = false;
    }
    int trainIdx = 0, testIdx = 0;

    // На обучение
    while (testIdx < testSize) {
        int idx = rand() % N;
        if (!used[idx]) {
            for (int j = 0; j <= M; j++) test[testIdx][j] = a[idx][j];
            used[idx] = true;
            testIdx++;
        }
    }

    // На тесты
    for (int i = 0; i < N; i++) {
        if (!used[i]) {
            for (int j = 0; j <= M; j++) train[trainIdx][j] = a[i][j];
            trainIdx++;
        }
    }

    // Учим
    // m[c][j] — среднее j-го признака для класса c
    // d[c][j] — СКО j-го признака для класса c
    // n[c]    — количество объектов класса c в train
    float** m = new float* [K];
    float** d = new float* [K];
    int* n = new int[K];

    for (int c = 0; c < K; c++) {
        m[c] = new float[M];
        d[c] = new float[M];
        n[c] = 0;
        for (int j = 0; j < M; j++) {
            m[c][j] = 0;
            d[c][j] = 0;
        }
    }

    // Суммы признаков по классам
    for (int i = 0; i < trainIdx; i++) {
        int c = (int)train[i][M];
        if (c >= 0 && c < K) {
            n[c]++;
            for (int j = 0; j < M; j++) {
                m[c][j] += train[i][j];
            }
        }
    }

    // Средние
    for (int c = 0; c < K; c++) {
        if (n[c] > 0) {
            for (int j = 0; j < M; j++) {
                m[c][j] /= n[c];
            }
        }
    }

    // Суммы квадратов отклонений
    float** q = new float* [K];
    for (int c = 0; c < K; c++) {
        q[c] = new float[M];
        for (int j = 0; j < M; j++) q[c][j] = 0;
    }

    for (int i = 0; i < trainIdx; i++) {
        int c = (int)train[i][M];
        if (c >= 0 && c < K && n[c] > 1) {
            for (int j = 0; j < M; j++) {
                float diff = train[i][j] - m[c][j];
                q[c][j] += diff * diff;
            }
        }
    }


    // СКО 
    for (int c = 0; c < K; c++) {
        for (int j = 0; j < M; j++) {
            if (n[c] > 1) {
                d[c][j] = sqrt(q[c][j] / (n[c] - 1));
            }
            else {
                d[c][j] = 1.0;   // защита от деления на 0
            }
            if (d[c][j] < 0.0001) d[c][j] = 0.0001;   // защита от нулевого СКО
        }
    }

    // Вывод статистик (первые 10 признаков, иначе много)
    for (int c = 0; c < K; c++) {
        cout << "Класс " << c << ": n=" << n[c] << endl;
        cout << "  Средние (первые 10): ";
        for (int j = 0; j < M && j < 10; j++) cout << m[c][j] << " ";
        if (M > 10) cout << "...";
        cout << endl;
        cout << "  СКО (первые 10):     ";
        for (int j = 0; j < M && j < 10; j++) cout << d[c][j] << " ";
        if (M > 10) cout << "...";
        cout << endl << endl;
    }

    // Классификация
    float* p = new float[K];
    float* post = new float[K];

    // Проверка на обучающей выборке
    int okTrain = 0;
    for (int i = 0; i < trainIdx; i++) {
        int cls = (int)train[i][M];

        for (int c = 0; c < K; c++) {
            float prod = 1.0;
            for (int j = 0; j < M; j++) {
                float coef = 1.0 / (sqrt(2 * 3.14159) * d[c][j]);
                float diff = train[i][j] - m[c][j];
                float e = -(diff * diff) / (2 * d[c][j] * d[c][j]);
                prod *= coef * exp(e);
            }
            p[c] = prod;
        }

        for (int c = 0; c < K; c++) {
            post[c] = ((float)n[c] / trainIdx) * p[c];
        }

        int pred = 0;
        float maxp = post[0];
        for (int c = 1; c < K; c++) {
            if (post[c] > maxp) { maxp = post[c]; pred = c; }
        }

        if (pred == cls) okTrain++;
    }

    cout << "Корректно на обучении: " << okTrain << "/" << trainIdx << endl;
    cout << "Точность на обучении: " << (float)okTrain / trainIdx * 100 << "%" << endl << endl;

    // Проверка на тестовой выборке
    int okTest = 0;
    for (int i = 0; i < testIdx; i++) {
        int cls = (int)test[i][M];

        for (int c = 0; c < K; c++) {
            float prod = 1.0;
            for (int j = 0; j < M; j++) {
                float coef = 1.0 / (sqrt(2 * 3.14159) * d[c][j]);
                float diff = test[i][j] - m[c][j];
                float e = -(diff * diff) / (2 * d[c][j] * d[c][j]);
                prod *= coef * exp(e);
            }
            p[c] = prod;
        }

        for (int c = 0; c < K; c++) {
            post[c] = ((float)n[c] / trainIdx) * p[c];
        }

        int pred = 0;
        float maxp = post[0];
        for (int c = 1; c < K; c++) {
            if (post[c] > maxp) { maxp = post[c]; pred = c; }
        }

        if (pred == cls) okTest++;
    }

    cout << "Корректно на тесте: " << okTest << "/" << testIdx << endl;
    cout << "Точность на тесте: " << (float)okTest / testIdx * 100 << "%" << endl;

    // Очистка памяти
    for (int i = 0; i < N; i++) delete[] a[i];
    delete[] a;

    for (int i = 0; i < trainSize; i++) delete[] train[i];
    delete[] train;

    for (int i = 0; i < testSize; i++) delete[] test[i];
    delete[] test;

    delete[] used;

    for (int c = 0; c < K; c++) {
        delete[] m[c];
        delete[] d[c];
        delete[] q[c];
    }
    delete[] m;
    delete[] d;
    delete[] q;
    delete[] n;
    delete[] p;
    delete[] post;

    return 0;

}