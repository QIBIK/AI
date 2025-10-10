namespace MO31_2_Myasoedov_Andrew.NeuroNet
{

    enum MemoryMode // режимы работы памяти
    {
        GET, // считывание памяти
        SET, // сохранение памяти
        INIT // инициализация памяти
    }

    enum NeuronType // тип нейрона
    {
        Hidden, // скрытый
        Output // выходной
    }

    enum NetworkMode // режим работы сети
    {
        Train, // обучение
        Test, // проверка
        Demo // распознование
    }
}

