using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    class OutputLayer : Layer
    {
        public OutputLayer(int non, int nopn, NeuronType nt, string type)
            : base(non, nopn, nt, type) { }

        // прямой проход
        public override void Recognize(Network net, Layer nextLayer)
        {
            double max = neurons.Max(n => n.Output);

            double[] expVals = new double[numofneurons];
            double sumExp = 0;

            for (int i = 0; i < numofneurons; i++)
            {
                expVals[i] = Math.Exp(neurons[i].Output - max); // стабилизированная softmax
                sumExp += expVals[i];
            }

            for (int i = 0; i < numofneurons; i++)
                net.Fact[i] = expVals[i] / sumExp;
        }


        // обратный проход
        public override double[] BackwardPass(double[] errors)
        {
            double[] gr_sum = new double[numofprevneurons + 1];
            for (int j = 0; j < numofprevneurons + 1; j++) // вычисление градиентных сумм выходного слоя
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                    sum += neurons[k].Weights[j] * errors[k];

                gr_sum[j] = sum;
            }

            for (int i = 0; i < numofneurons; i++) // цикл коррекции синаптических весов
                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0) // если порог
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * errors[i];
                    else
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * errors[i];

                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw; // коррекция весов
                }

                return gr_sum;
        }
    }
}
