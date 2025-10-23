using System;
using System.IO;
using System.Reflection.Emit;
using System.Windows.Forms;

namespace MO31_2_Myasoedov_Andrew.NeuroNet
{
    abstract class Layer
    {
        // Поля
        protected string name_Layer; // наименование слоя, которое используется
        string pathDirWeights; // путь к каталогу, где находится файл
        string pathFileWeights; // путь к файлу синаптических весов
        protected int numofneurons; // число нейронов текущего слоя
        protected int numofprevneurons; // число нейронов предыдущего слоя
        protected const double learningrate = 0.060; // скорость обучения
        protected const double momentum = 0.050d; // момент инерции
        protected double[,] lastdeltaweights; // веса предыдущей итерации
        protected Neuron[] neurons; // массив нейроново текущего

        // Свойства
        public Neuron[] Neurons { get => neurons; set => neurons = value; }
        public double[] Data // передача входных данных на нейроны слоя и активации
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                {
                    Neurons[i].Activator(value);
                }
            }
        }

        // Конструктор
        protected Layer(int non, int nopn, NeuronType nt,  string nm_Layer)
        {
            int i, j; // счётчики циклов
            numofneurons = non; // количесвто нейронов текущего слоя
            numofprevneurons = nopn; // количество нейронов предыдущего слоя
            Neurons = new Neuron[non]; // определение массива нейронов
            name_Layer = nm_Layer; // наименование слоя, котоое используется
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "memory.csv";

            lastdeltaweights = new double[non, nopn + 1];
            double[,] Weights; // временный массив синаптических весов текущего

            if (File.Exists(pathFileWeights)) // определяет, сущесвтуеют ли pathFileWeights
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights);
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (i = 0; i < non; i++) // цикл формирования нейронов слоя и заполнения
            {
                double[] tmp_weights = new double[nopn + 1];
                for (j = 0;  j < nopn + 1; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt); // заполнение массива нейронами
            }
        }

        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            int i, j;
            char[] delim = new char[] { ';', ' ' };
            string tmpStr;
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];
        
            switch (mm)
            {
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path); // считываение строк текстового
                    string[] memory_element;
                    for (i = 0; i < numofneurons; i++)
                    {
                        memory_element = tmpStrWeights[i].Split(delim); // разбивает строку
                        for (j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_element[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                case MemoryMode.SET:
                    tmpStr = "";
                    for (i = 0; i < numofneurons; i++)
                    {
                        string[] tmpRow = new string[numofprevneurons + 1];
                        for (j = 0; j < numofprevneurons + 1; j++)
                        {
                            tmpRow[j] = Neurons[i].Weights[j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture);
                        }
                        tmpStr += string.Join(";", tmpRow) + "\n";
                    }
                    File.WriteAllText(path, tmpStr);
                    break;

                case MemoryMode.INIT:
                    Random random = new Random();
                    for (i = 0; i < numofneurons; i++)
                    {
                        double sum = 0.0;
                        double squaredSum = 0.0;

                        // Генерация случайных весов [-1; +1]
                        for (j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = random.NextDouble() * 2.0 - 1.0;
                            sum += weights[i, j];
                            squaredSum += weights[i, j] * weights[i, j];
                        }

                        // Нормализация весов (среднее = 0, σ = 1)
                        double mean = sum / (numofprevneurons + 1);
                        double variance = (squaredSum / (numofprevneurons + 1)) - (mean * mean);
                        double root = Math.Sqrt(Math.Max(variance, 1e-8)); // защита от деления на 0

                        for (j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = (weights[i, j] - mean) / root;
                        }
                    }

                    // Сохранение весов в CSV
                    string[] lines = new string[numofneurons];
                    for (i = 0; i < numofneurons; i++)
                    {
                        string[] row = new string[numofprevneurons + 1];
                        for (j = 0; j < numofprevneurons + 1; j++)
                        {
                            row[j] = weights[i, j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture)
                                .Replace('.', ',');
                        }
                        lines[i] = string.Join(";", row);
                    }
                    File.WriteAllLines(path, lines);
                    break;
            }
            return weights;
        }
        abstract public void Recognize(Network net, Layer nextLayer); //для прямых проходов
        abstract public double[] BackwardPass(double[] stuff); //и обратных
    }
}


//как генерируется синаптические веса
// Синаптические веса должны быть случайными значениями от -1 до +1
// У каждого нейрона синаптические веса и порог, среднее мат ожидание должно быть = 0
// Среднее квадратичное отклонение должно быть = 1

