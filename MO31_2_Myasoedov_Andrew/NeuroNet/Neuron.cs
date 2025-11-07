using static System.Math;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    class Neuron
    {
        private NeuronType type; // тип нейрона
        private double[] weights; // его веса
        private double[] inputs; // его входы
        private double output; // его выходы
        private double derivative; // производная

        // свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        // конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }


        public void Activator(double[] i)
        {
            inputs = i; // передача вектора входного сигнала в массив входных данных нейрона
            double sum = weights[0]; // кладём первый элемент в сумму

            for (int j = 0; j < inputs.Length; j++) // цикл вычисления индуцированного поля нейрона
            {
                sum += inputs[j] * weights[j + 1]; // линейное преобразование входных сигналов 
            }

            switch (type)
            {
                case NeuronType.Hidden:  // для нейронов скрытого слоя
                    output = HyperbolicTangent(sum);
                    derivative = DerivativeHyperbolicTangent(output);
                    break;

                case NeuronType.Output: // для нейронов выходного слоя
                    output = Exp(sum);
                    break;
            }
        }

        // функция активации каждый пишет сам

        private double HyperbolicTangent(double sum) // для гиперболического тангенса
        {
            double expon1 = Exp(sum);
            double expon2 = Exp(-sum);
            double outputTang = (expon1 - expon2) / (expon1 + expon2); // tan(x) = (e^x - e^(-x))/(e^x + e^(-x))
            return outputTang; // возвращаем значение
        }


        // производная гиперболического тангенса
        private double DerivativeHyperbolicTangent(double output)
        {
            return 1 - (output * output); // f'(x) = 1 - f(x)^2
        }


    }
}