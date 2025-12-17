using System;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Windows.Forms.DataVisualization.Charting;
using MO31_2_Myasoedov_Andrew.NeuroNet;

namespace MO31_2_Myasoedov_Andrew
{
    public partial class Form1 : Form
    {
        private Layer outputLayer;
        private Layer hiddenLayer;
        private double[] inputPixels; // массив входных данных
        private Network network; // объявление нейросети

        //Конструктор
        public Form1()
        {
            InitializeComponent();

            inputPixels = new double[15]; //Количество пикселей
            network = new Network();
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

        private void button_Recognize_Click(object sender, EventArgs e)
        {
            network.ForwardPass(network, inputPixels);
            label_out.Text = network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            label_probability.Text = (100 * network.Fact.Max()).ToString("0.00") + " %";
        }
        private void button_Train_Click(object sender, EventArgs e)
        {

            network.Train(network);
            for (int i = 0; i < network.E_errors_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_errors_avr[i]);
                chart_Accuracy.Series[0].Points.AddY(network.Accuracy_avr[i]);
            }

            MessageBox.Show("Обучение успешно завершено.", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void button_Test_Click(object sender, EventArgs e)
        {
            network.Test(network);

            for (int i = 0; i < network.E_errors_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_errors_avr[i]);
                chart_Accuracy.Series[0].Points.AddY(network.Accuracy_avr[i]);
            }

            MessageBox.Show("Тестирование успешно завершено.", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void chart1_Click(object sender, EventArgs e){}
        private void chart_Eavr_Click(object sender, EventArgs e){}
    }
}
