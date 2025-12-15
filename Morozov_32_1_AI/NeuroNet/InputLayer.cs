using System;
using System.IO;
using Morozov_32_1_AI.NeuroNet;

namespace Morozov_32_1_AI.NeuroNet
{
    internal class InputLayer
    {
        private double[,] trainset;
        private double[,] testset;

        public double[,] Trainset { get => trainset; }
        public double[,] Testset { get => testset; }

        public InputLayer(NetworkMode nm)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory;
            string[] tmpArrStr;
            string[] tmpStr;

            switch(nm)
            {
                case NetworkMode.Train:
                    tmpArrStr = File.ReadAllLines(path + "train.txt");
                    trainset = new double[tmpArrStr.Length, 16];

                    for(int i = 0; i < tmpArrStr.Length;i++)
                    {
                        tmpStr = tmpArrStr[i].Split(' ');

                        for(int j = 0; j < 16; j++)
                        {
                            trainset[i, j] = double.Parse(tmpStr[j]);
                        }
                    }
                    Shuffling_Array_Rows(trainset);
                    break;
                case NetworkMode.Test:
                    tmpArrStr = File.ReadAllLines(path + "test.txt");
                    testset = new double[tmpArrStr.Length, 16];

                    for (int i = 0; i < tmpArrStr.Length; i++)
                    {
                        tmpStr = tmpArrStr[i].Split(' ');

                        for (int j = 0; j < 16; j++)
                        {
                            testset[i, j] = double.Parse(tmpStr[j]);
                        }
                    }
                    Shuffling_Array_Rows(testset);
                    break;
            }
        }
        public void Shuffling_Array_Rows(double[,] arr)
        {
            int numRows = arr.GetLength(0); // Получаем количество строк
            int numCols = arr.GetLength(1); // Получаем количество столбцов (чтобы знать размер строки)

            Random random = new Random(); // Создаем генератор случайных чисел

            // Итерация с конца до начала массива
            for (int i = numRows - 1; i >= 0; i--)
            {
                // Выбираем случайный индекс j от 0 до i (включительно)
                int j = random.Next(i + 1);

                // Теперь нужно поменять местами строку i и строку j
                // Для этого придется временно сохранить строку i
                double[] tempRow = new double[numCols];

                // Копируем строку i во временный буфер
                for (int col = 0; col < numCols; col++)
                {
                    tempRow[col] = arr[i, col];
                }

                // Копируем строку j в строку i
                for (int col = 0; col < numCols; col++)
                {
                    arr[i, col] = arr[j, col];
                }

                // Копируем временный буфер (старую строку i) в строку j
                for (int col = 0; col < numCols; col++)
                {
                    arr[j, col] = tempRow[col];
                }
            }
        }
    }
}
