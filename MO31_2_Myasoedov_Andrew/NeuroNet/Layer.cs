using System;
using System.Collections.Generic;
using System.Configuration;
using System.Drawing;
using System.IO;
using System.Linq;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    abstract class Layer
    {
        protected string name_Layer; // название слоя
        string pathDirWeights; // путь к каталогу, где находится файл синаптических весов
        string pathFileWeights; // путь к файлу саниптическов весов
        protected int numofneurons; // число нейронов текущего слоя
        protected int numofprevneurons; // число нейронов предыдущего слоя
        protected const double learningrate = 0.0265; // скорость обучения 0.06
        protected const double momentum = 0.011d; // момент инерции 0.050d
        protected double[,] lastdeltaweights; // веса предыдущей итерации
        protected Neuron[] neurons; // массив нейронов текущего слоя

        // свойства
        public Neuron[] Neurons { get => neurons; set => neurons = value; }

        // активация нейрона
        public double[] Data // передача входных сигналов на нейроны слоя и активатор
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                    neurons[i].Activator(value);
            }
        }

        // конструктор
        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer)
        {
            numofneurons = non; // количество нейронов текущего слоя
            numofprevneurons = nopn; // количество нейронов предыдущего слоя
            neurons = new Neuron[non]; // определение массива нейронов
            name_Layer = nm_Layer; // наиминование слоя
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";

            double[,] Weights; // временный массив синаптических весов
            lastdeltaweights = new double[non, nopn + 1];

            if (File.Exists(pathFileWeights)) // определяет существует ли pathFileWeights
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights); //считывает данные из файла
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (int i = 0; i < non; i++) // цикл формирования нейронов слоя и заполнения
            {
                double[] tmp_weights = new double[nopn + 1];
                for (int j = 0; j < nopn; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                neurons[i] = new Neuron(tmp_weights, nt); // заполнение массива нейронами
            }
        }


        private double[] InitializeWeightsWithConstraints(int count, Random random)
        {
            double[] rawWeights = new double[count];

            // генерируем веса в [-1, 1]
            for (int j = 0; j < count; j++)
            {
                rawWeights[j] = random.NextDouble() * 2.0 - 1.0;
            }

            // корректируем, чтобы среднее стало 0
            double mean = rawWeights.Average();
            for (int j = 0; j < count; j++)
            {
                rawWeights[j] -= mean;
            }

            // вычисляем текущий stddev
            double variance = rawWeights.Sum(w => w * w) / count; // since mean is now 0
            double currentStdDev = Math.Sqrt(variance);

            // если текущий stddev > 0, масштабируем к stddev = 1
            if (currentStdDev > 0)
            {
                double scale = 1.0 / currentStdDev;
                for (int j = 0; j < count; j++)
                {
                    rawWeights[j] *= scale;
                }
            }

            // проверяем, не вышли ли веса за [-1, 1] и корректируем, если нужно
            double maxAbs = rawWeights.Max(Math.Abs);
            if (maxAbs > 1.0)
            {
                // сжимаем все веса чтобы максимальный по модулю был 1
                double rescale = 1.0 / maxAbs;
                for (int j = 0; j < count; j++)
                {
                    rawWeights[j] *= rescale;
                }

                // после сжатия среднее может снова сдвинуться, корректируем его
                double finalMean = rawWeights.Average();
                for (int j = 0; j < count; j++)
                {
                    rawWeights[j] -= finalMean;
                }
            }

            return rawWeights;
        }

        // метод работы с массивом синаптических весов слоя
        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' };
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            switch (mm)
            {
                // парсинг в тип double строкового формата веса нейронов из csv - получает значения весов нейронов
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path);        // считывание строк текстового файла csv весов нейрона (в tmpStrWeights каждый i-ый элемент это строка весов)
                    string[] memory_elemnt; // массив, где каждый i-ый элемент это один вес нейрона (берётся одна строка из tmpStrWeights)

                    // строка весов нейронов
                    for (int i = 0; i < numofneurons; i++)
                    {
                        memory_elemnt = tmpStrWeights[i].Split(delim);  // разбивает строку
                        // каждый отдельный вес нейрона
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_elemnt[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                // парсинг в строковой формат веса нейронов в csv (обратный MemoryMode.GET) - сохраняет готовые веса нейронов
                case MemoryMode.SET:
                    tmpStrWeights = new string[numofneurons]; // создаём строку из весов нейрона (tmpStrWeights это массив, где каждый i-ый элемент это строка весов) 
                    for (int i = 0; i < numofneurons; i++)
                    {
                        string[] memory_elemnt2 = new string[numofprevneurons + 1];
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            memory_elemnt2[j] = neurons[i].Weights[j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture)
                                .Replace('.', ',');
                        }
                        tmpStrWeights[i] = string.Join(";", memory_elemnt2);
                    }
                    File.WriteAllLines(path, tmpStrWeights);
                    break;

                // инициализация весов для нейронов
                case MemoryMode.INIT:
                    tmpStrWeights = new string[numofneurons];
                    Random random = new Random();

                    for (int i = 0; i < numofneurons; i++)
                    {
                        int count = numofprevneurons + 1;
                        double[] rawWeights = InitializeWeightsWithConstraints(count, random);

                        // сохраняем результат в массив weights
                        for (int j = 0; j < count; j++)
                        {
                            weights[i, j] = rawWeights[j];
                        }

                        // запись в файл
                        string[] memory_elemnt2 = new string[count];
                        for (int j = 0; j < count; j++)
                        {
                            memory_elemnt2[j] = weights[i, j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture)
                                .Replace('.', ',');
                        }
                        tmpStrWeights[i] = string.Join(";", memory_elemnt2);
                    }

                    File.WriteAllLines(path, tmpStrWeights);
                    break;

            }
            return weights;
        }

        abstract public void Recognize(Network net, Layer nextLayer); // для прямых проходов

        abstract public double[] BackwardPass(double[] stuff); // и их обратных
    }
}