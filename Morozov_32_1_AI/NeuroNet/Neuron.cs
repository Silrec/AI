using System;
using static System.Math;

namespace Morozov_32_1_AI.NeuroNet
{
    class Neuron
    {
        // поля
        private NeuronType type; //тип нейрона
        private double[] weights; //его веса
        private double[] inputs; //его входы
        private double output;
        private double derivative;

        //константы для функц активации
        private double a = 0.01d;

        //свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        //конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }
        ///*
        public void Activator(double[] i)
        {
            inputs = i;
            double sum = weights[0];
            for (int j = 0; j < inputs.Length; j++)
            {
                sum += inputs[j] * weights[j + 1];
            }

            switch (type)
            {
                case NeuronType.Hidden:
                    output = 1.0 / (1.0 + Math.Exp(-sum));
                    derivative = output * (1.0 - output);
                    break;

                case NeuronType.Output:
                    output = Math.Exp(sum);
                    break;
            }
        }
        //*/
        /*
        public void Activator(double[] i)
        {
            inputs = i;
            double sum = weights[0];
            for (int j = 0; j < inputs.Length; j++)
            {
                sum += inputs[j] * weights[j + 1];
            }

            switch (type)
            {
                case NeuronType.Hidden:
                    // Leaky ReLU activation
                    if (sum >= 0)
                    {
                        output = sum;
                        derivative = 1.0;
                    }
                    else
                    {
                        output = 0.01 * sum;  // коэффициент 0.01 для отрицательных значений
                        derivative = 0.01;
                    }
                    break;

                case NeuronType.Output:
                    output = Math.Exp(sum);
                    break;
            }
        }
        */

    }

}
