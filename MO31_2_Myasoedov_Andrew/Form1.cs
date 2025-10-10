using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace MO31_2_Myasoedov_Andrew
{
    public partial class Form1 : Form
    {
        private double[] inputPixels;

        //Конструктор
        public Form1()
        {
            InitializeComponent();

            inputPixels = new double[15]; //Количество пикселей

            // создаём нейросеть при запуске программы
            var net = new MO31_2_Myasoedov_Andrew.NeuroNet.Network();
        }

        private void Changing_State_Pixel_Button_Click(object sender, EventArgs e)
        {
            if (((Button)sender).BackColor == Color.Black)
            {
                ((Button)sender).BackColor = Color.White;
                inputPixels[((Button)sender).TabIndex] = 1d; //1d = 1.0
            }
            else
            {
                ((Button)sender).BackColor = Color.Black;
                inputPixels[((Button)sender).TabIndex] = 0d; //0d = 0.0
            }
        }

        private void button_SaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";
            string tmpStr = numericUpDownNecessary.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n"; // переход на новую строку текста

            File.AppendAllText(path, tmpStr); // добавление текста tmpStr
                                              // в файл, расположенный по path
        }

        private void button_SaveTestSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "test.txt";
            string tmpStr = numericUpDownNecessary.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n"; // переход на новую строку текста

            File.AppendAllText(path, tmpStr); // добавление текста tmpStr
                                              // в файл, расположенный по path
        }
    }
}
