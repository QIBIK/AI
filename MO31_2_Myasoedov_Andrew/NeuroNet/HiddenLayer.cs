using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    internal class HiddenLayer : Layer
    {
        public HiddenLayer(int numOfNeurons, int numOfPrevNeurons, string name)
            : base(numOfNeurons, numOfPrevNeurons, NeuronType.Hidden, name)
        {
        }
    }
}
