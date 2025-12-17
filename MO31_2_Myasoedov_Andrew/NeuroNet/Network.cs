using System;
using System.Linq;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    class Network
    {
        // все слои сети
        private InputLayer input_layer = null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(70, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(35, 70, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 35, NeuronType.Output, nameof(output_layer));

        private double[] fact = new double[10]; // массив фактического выхода сети
        private double[] e_error_avr; // среднее значение энергии ошибки
        private double[] accuracy_avr; // для точности
        // свойства
        public double[] Fact { get => fact; } // массив фактического выхода сети
        // среднее значение энергии ошибки эпохи обучения
        public double[] E_errors_avr { get => e_error_avr; set => e_error_avr = value; }
        public double[] Accuracy_avr { get => accuracy_avr; set => accuracy_avr = value; }
        // конструктор
        public Network() { }

        public void ForwardPass(Network net, double[] netInput)
        {
            net.hidden_layer1.Data = netInput;
            net.hidden_layer1.Recognize(null, net.hidden_layer2);
            net.hidden_layer2.Recognize(null, net.output_layer);
            net.output_layer.Recognize(net, null);
        }

        // Метод обучения
        public void Train(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Train); // инициализация входного слоя
            int epoches = 10;
            double tmpSumError; // временная переменная суммы ошибок
            double[] errors; // вектор сигнала ошибки выходного слоя
            double[] temp_gsums1; // вектор градиента 1-огго скрытого слоя
            double[] temp_gsums2;

            e_error_avr = new double[epoches];
            accuracy_avr = new double[epoches]; // добавляем массив для точности

            for (int k = 0; k < epoches; k++) // перебор эпох обучения
            {
                e_error_avr[k] = 0; // вначале каждой жпохи обучения значение средней энергии ошибки эпохи
                accuracy_avr[k] = 0; // точность на эпохе

                net.input_layer.Shuffling_Array_Rows(net.input_layer.Trainset); // перетасовка обучаюзей выборки
                int correctPredictions = 0;

                for (int i = 0; i < net.input_layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15];  //обучающий образ
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1]; // 

                    // прямой проход
                    ForwardPass(net, tmpTrain); // прямой проход обучающего образа

                    //вычисление ошибки
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Trainset[i, 0])
                            errors[x] = 1.0 - net.Fact[x];
                        else
                            errors[x] = -net.fact[x];
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; // Суммарное значение энергии ошибки

                    // вычисление точности
                    int predictedClass = Array.IndexOf(net.Fact, net.Fact.Max());
                    int trueClass = (int)net.input_layer.Trainset[i, 0];
                    if (predictedClass == trueClass)
                        correctPredictions++;

                    // обратный проход и коррекция весов !!!!!
                    temp_gsums2 = net.output_layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= net.input_layer.Trainset.GetLength(0);
                accuracy_avr[k] = (double)correctPredictions / net.input_layer.Trainset.GetLength(0);

            }

            net.input_layer = null;

            net.hidden_layer1.WeightInitialize(MemoryMode.SET, nameof(hidden_layer1) + "_memory.csv");
            net.hidden_layer2.WeightInitialize(MemoryMode.SET, nameof(hidden_layer2) + "_memory.csv");
            net.output_layer.WeightInitialize(MemoryMode.SET, nameof(output_layer) + "_memory.csv");
        }


        /*public void Training(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Train); // инициализация входного слоя
            int epoches = 20;
            double tmpSumError; // временная переменная суммы ошибок
            double[] errors; // вектор сигнала ошибки выходного слоя
            double[] temp_gsums1; // вектор градиента 1-огго скрытого слоя
            double[] temp_gsums2;

            e_error_avr = new double[epoches];

            for (int k = 0; k < epoches; k++) // перебор эпох обучения
            {
                e_error_avr[k] = 0; // вначале каждой жпохи обучения значение средней энергии ошибки эпохи
                net.input_layer.Shuffling_Array_Rows(net.input_layer.Trainset); // перетасовка обучаюзей выборки
                for (int i = 0; i < net.input_layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15];  //обучающий образ
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1]; // 

                    // прямой проход
                    ForwardPass(net, tmpTrain); // прямой проход обучающего образа

                    //вычмсление ошибки
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Trainset[i, 0])
                            errors[x] = 1.0 - net.Fact[x];
                        else
                            errors[x] = -net.fact[x];
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; // Суммарное значение энергии ошибки

                    // обратный проход и коррекция весов !!!!!
                    temp_gsums2 = net.output_layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= net.input_layer.Trainset.GetLength(0);
            }

            net.input_layer = null;

        }*/

        // для теста (новая)
        public void Test(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Test); // инициализация входного слоя
            int epoches = 10; // эпохи обучения
            double tmpSumError; // временная переменная суммы ошибок
            double[] errors; // вектор сигнала ошибки выходного слоя

            e_error_avr = new double[epoches];
            accuracy_avr = new double[epoches]; // добавляем массив для точности


            for (int k = 0; k < epoches; k++) // перебор эпох обучения
            {
                e_error_avr[k] = 0; // вначале каждой эпохи обучения значение средней энергии ошибки эпохи
                accuracy_avr[k] = 0; // точность на эпохе

                net.input_layer.Shuffling_Array_Rows(net.input_layer.Testset); // перетасовка обучающей выборки
                int correctPredictions = 0;

                for (int i = 0; i < net.input_layer.Testset.GetLength(0); i++)
                {
                    double[] tmpTest = new double[15];  //обучающий образ
                    for (int j = 0; j < tmpTest.Length; j++)
                        tmpTest[j] = net.input_layer.Testset[i, j + 1]; // 

                    // прямой проход
                    ForwardPass(net, tmpTest); // прямой проход обучающего образа

                    //вычисление ошибки
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Testset[i, 0])
                            errors[x] = 1.0 - net.Fact[x];
                        else
                            errors[x] = -net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; // суммарное значение энергии ошибки

                    // вычисление точности
                    int predictedClass = Array.IndexOf(net.Fact, net.Fact.Max());
                    int trueClass = (int)net.input_layer.Testset[i, 0];
                    if (predictedClass == trueClass)
                        correctPredictions++;

                }
                e_error_avr[k] /= net.input_layer.Testset.GetLength(0);
                accuracy_avr[k] = (double)correctPredictions / net.input_layer.Testset.GetLength(0);

            }



        }
    }
}