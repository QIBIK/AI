namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_Layer) : base(non, nopn, nt, nm_Layer)
        {

        }

        // прямой подход
        public override void Recognize(Network net, Layer nextLayer)
        {
            double[] hidden_out = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
                hidden_out[i] = neurons[i].Output;

            nextLayer.Data = hidden_out; // передача выходного сигнала на вход следующего слоя
        }

        // обратный проход
        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];
            for (int j = 0; j < numofprevneurons; j++) // цикл вычисления градиента (сумма j-ого нейрона)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                {
                    sum += neurons[k].Weights[j+1] * neurons[k].Derivative * gr_sums[k]; // через градиентные суммы и производные
                }
                gr_sum[j] = sum;
            }


            for (int i = 0; i < numofneurons; i++)
            {
                for (int n = 0; n < numofprevneurons; n++)
                {
                    double delwat;
                    if (n == 0) // если порог
                        delwat = momentum * lastdeltaweights[i, 0] + learningrate * neurons[i].Derivative * gr_sums[i];
                    else
                        delwat = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * neurons[i].Derivative * gr_sums[i];

                    lastdeltaweights[i, n] = delwat;
                    neurons[i].Weights[n] += delwat; // коррекция весов
                }
            }
            return gr_sum;
        }
    }
}