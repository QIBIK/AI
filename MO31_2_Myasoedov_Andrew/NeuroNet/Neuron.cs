using System;
using System.Diagnostics.SymbolStore;
using static System.Math;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    class Neuron
    {
        // поля
        private NeuronType type; // тип нейрона
        private double[] weights; // его веса
        private double[] inputs; // его входы
        private double output; // его выход
        private double derivative; // производная

        // константы для функции активации
        private double a = 0.01d;

        // свойства
        public double[]  Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        // конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }

        // метод активации нейрона
        public void Activator(double[] i)
        {
            inputs = i; // передача вектора входного сигнала в массив входных данных нейрона
            
            double sum = weights[0]; // аффиное преобразование через смещение (нулевой вес 

            for (int j = 0; j < inputs.Length; j++) // цикл вычисления индуцированного поля нейрона
            {
                sum += inputs[j] * weights[j + 1]; // линейные преобразования входных сигналов
            }

            switch (type)
            {
                case NeuronType.Hidden: // для нейронов скрытого слоя
                    output = Tanh(sum);
                    derivative = Tanh_Derivativator(sum);
                    break;

                case NeuronType.Output: // для нейронов выходного слоя
                    output = Tanh(sum);
                    derivative = Tanh_Derivativator(sum);
                    break;
            }
        }

        // Гиперболический тангенс
        private double Tanh(double x)
        {
            // чтобы избежать переполнения при больших x
            if (x > 20) return 1;
            if (x < -20) return -1;

            double ex = Math.Exp(x);
            double enx = Math.Exp(-x);
            return (ex - enx) / (ex + enx);
        }

        // Производная гиперболического тангенса
        private double Tanh_Derivativator(double x)
        {
            double t = Tanh(x);
            return 1 - t * t;
        }
    }
}

