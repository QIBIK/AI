using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    internal class OutputLayer : Layer
    {
        public OutputLayer(int numOfNeurons, int numOfPrevNeurons, string name)
            : base(numOfNeurons, numOfPrevNeurons, NeuronType.Output, name)
        {
        }
    }
}
