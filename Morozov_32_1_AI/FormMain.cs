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
using Morozov_32_1_AI.NeuroNet;

namespace Morozov_32_1_AI
{

    public partial class FormMain : Form
    {
        private Layer hiddenLayer;
        private Layer outputLayer;
        private Network network;


        private double[] inputPixels;

        //конструктор
        public FormMain()
        {
            InitializeComponent();
            inputPixels = new double[15];

            network = new Network();
        }

        //обработка пикселя
        private void Changing_State_Pixel_Button_Click(object sender, EventArgs e)
        {
            if (((Button)sender).BackColor == Color.White)
            {
                ((Button)sender).BackColor = Color.Black;
                inputPixels[((Button)sender).TabIndex] = 1d;
            }
            else
            {
                ((Button)sender).BackColor = Color.White;
                inputPixels[((Button)sender).TabIndex] = 0d;
            }
        }

        //сохранение примера
        private void button_SaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr);
        }

        private void button_SaveTestSample_Click_1(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "test.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr);
        }

        private void recognizebutton_Click(object sender, EventArgs e)
        {
            network.ForwardPass(network, inputPixels, false);

            label_Output.Text = network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            label_probability.Text = (100 * network.Fact.Max()).ToString("0.00") + " %";
        }

        private void button_training_Click(object sender, EventArgs e)
        {
            network.SetDropoutStatus(checkBox_UseDropout.Checked);
            network.Train(network);
            
            for(int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series["Series_Eavr"].Points.AddY(network.E_error_avr[i]);
                chart_Eavr.Series["Series_Acc"].Points.AddY(network.Accuracy_avr[i]);
            }
            MessageBox.Show("Обучение завершено", "Info");
        }

        private void label_probability_Click(object sender, EventArgs e)
        {

        }

        private void FormMain_Load(object sender, EventArgs e)
        {

        }

        private void chart1_Click(object sender, EventArgs e)
        {

        }

        private void button_Test_Click(object sender, EventArgs e)
        {
            
            network.Test(network);
            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]);
            }
            MessageBox.Show("Тестирование завершено", "Info");
        }

        private void label_Output_Click(object sender, EventArgs e)
        {

        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {

        }
    }
}
