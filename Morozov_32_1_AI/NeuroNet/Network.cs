using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Morozov_32_1_AI.NeuroNet;

namespace Morozov_32_1_AI.NeuroNet
{
    class Network
    {
        //все слои сети
        private InputLayer input_layer = null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(71, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(35, 71, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 35, NeuronType.Output, nameof(output_layer));

        private double[] fact = new double[10]; //массив фактического выхода сети
        private double[] e_error_avr; //среднее значение энергии ошибки эпохи обучения

        //Свойства
        public double[] Fact { get => fact; } //массив фактического выхода сети
        // среднее
        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }
        //Конструктор

        private double[] accuracy_avr;

        public double[] Accuracy_avr { get => accuracy_avr; }
        public Network() { }

        public void SetDropoutStatus(bool isActive)
        {
            double rate = isActive ? 0.20 : 0.0;

            hidden_layer1.DropoutRate = rate;
            hidden_layer2.DropoutRate = rate;

        }

        public void ForwardPass(Network net, double[] netInput, bool isTraining)
        {
            net.hidden_layer1.Data = netInput;
            net.hidden_layer1.Recognize(null, net.hidden_layer2, isTraining);
            net.hidden_layer2.Recognize(null, net.output_layer, isTraining);
            net.output_layer.Recognize(net, null, false);
        }
        public void Train(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Train);
            bool isDropoutActive = net.hidden_layer1.DropoutRate > 0;
            int epoches = isDropoutActive ? 100 : 10;

            Layer.learningrate = 0.5;

            double tmpSumError;
            double[] errors;
            double[] temp_gsums1;
            double[] temp_gsums2;

            e_error_avr = new double[epoches];
            accuracy_avr = new double[epoches];
            for (int k = 0; k< epoches; k++)
            {
                e_error_avr[k] = 0;
                accuracy_avr[k] = 0;
                int correct_predictions = 0;
                net.input_layer.Shuffling_Array_Rows(net.input_layer.Trainset);
                for (int i =0; i< net.input_layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15];
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1];

                    ForwardPass(net, tmpTrain, true);

                    double maxVal = net.fact[0];
                    int maxIdx = 0;
                    for (int n = 1; n < net.fact.Length; n++)
                    {
                        if (net.fact[n] > maxVal)
                        {
                            maxVal = net.fact[n];
                            maxIdx = n;
                        }
                    }
                    if (maxIdx == (int)net.input_layer.Trainset[i, 0])
                    {
                        correct_predictions++;
                    }

                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for(int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Trainset[i, 0])
                            errors[x] = 1.0 - net.fact[x];
                        else
                            errors[x] = -net.fact[x];
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length;

                    temp_gsums2 = net.output_layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= net.input_layer.Trainset.GetLength(0);

                accuracy_avr[k] = (double)correct_predictions / net.input_layer.Trainset.GetLength(0);
            }
            net.input_layer = null;
            net.hidden_layer1.WeightInitialize(MemoryMode.SET, nameof(hidden_layer1) + "_memory.csv");
            net.hidden_layer2.WeightInitialize(MemoryMode.SET, nameof(hidden_layer2) + "_memory.csv");
            net.output_layer.WeightInitialize(MemoryMode.SET, nameof(output_layer) + "_memory.csv");
        }
        public void Test(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Test);

            int epoches = 5;
            double tmpSumError;
            double[] errors;

            e_error_avr = new double[epoches];
            for (int k = 0; k < epoches; k++)
            {
                e_error_avr[k] = 0;
                net.input_layer.Shuffling_Array_Rows(net.input_layer.Testset);
                for (int i = 0; i < net.input_layer.Testset.GetLength(0); i++)
                {
                    double[] tmpTest = new double[15];
                    for (int j = 0; j < tmpTest.Length; j++)
                        tmpTest[j] = net.input_layer.Testset[i, j + 1];

                    ForwardPass(net, tmpTest, false);

                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Testset[i, 0])
                            errors[x] = 1.0 - net.fact[x];
                        else
                            errors[x] = -net.fact[x];
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length;
                }
                e_error_avr[k] /= net.input_layer.Testset.GetLength(0);
            }
            net.input_layer = null;
        }
    }
}
