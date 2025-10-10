using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    internal class Network
    {
        private HiddenLayer hiddenLayer1;
        private HiddenLayer hiddenLayer2;
        private OutputLayer outputLayer;

        public Network()
        {
            // Архитектура сети: 15 -> 70 -> 35 -> 10
            hiddenLayer1 = new HiddenLayer(70, 15, nameof(hiddenLayer1));
            hiddenLayer2 = new HiddenLayer(35, 70, nameof(hiddenLayer2));
            outputLayer = new OutputLayer(10, 35, nameof(outputLayer));
        }

        public double[] Run(double[] input)
        {
            hiddenLayer1.Data = input;
            double[] outHidden1 = hiddenLayer1.Neurons.Select(n => n.Output).ToArray();

            hiddenLayer2.Data = outHidden1;
            double[] outHidden2 = hiddenLayer2.Neurons.Select(n => n.Output).ToArray();

            outputLayer.Data = outHidden2;
            double[] outFinal = outputLayer.Neurons.Select(n => n.Output).ToArray();

            return outFinal;
        }
    }
}
