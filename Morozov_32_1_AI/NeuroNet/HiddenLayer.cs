using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Morozov_32_1_AI.NeuroNet;

namespace Morozov_32_1_AI.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_Layer) : base(non, nopn, nt, nm_Layer) { }

        //прямой проход
        public override void Recognize(Network net, Layer nextLayer, bool isTraining)
        {
            double[] hidden_out = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
            {
                hidden_out[i] = neurons[i].Output;
                if (isTraining && dropoutRate > 0)
                {
                    
                    if (random.NextDouble() >= dropoutRate)
                        dropoutMask[i] = 1.0;
                    else
                        dropoutMask[i] = 0.0;

                    
                    hidden_out[i] *= dropoutMask[i];

                    hidden_out[i] *= (1.0 / (1.0 - dropoutRate));
                }
                else
                {
                    dropoutMask[i] = 1.0;
                }
            }

            nextLayer.Data = hidden_out;
        }

        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];

            // Коэффициент усиления
            double scalingFactor = (dropoutRate > 0) ? (1.0 / (1.0 - dropoutRate)) : 1.0;

            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                {
                    
                    double influence = dropoutMask[k] * scalingFactor;

                    sum += neurons[k].Weights[j + 1] * neurons[k].Derivative * gr_sums[k]; //через градиентные суммы и производную

                }
                gr_sum[j] = sum;
            }

            for (int i = 0; i < numofneurons; i++) //цикл коррекции синаптических весов
            {
                // Если нейрон был выключен, мы НЕ меняем его веса
                if (dropoutMask[i] == 0.0) continue;

                // Усиливаем приходящий градиент, чтобы компенсировать пропуск
                double correctedGradient = gr_sums[i] * scalingFactor;

                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0) //если порог
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * neurons[i].Derivative * correctedGradient;
                    else
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * neurons[i].Derivative * correctedGradient;

                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw; //коррекция весов
                }
            }
            return gr_sum;
        }
    }
}
