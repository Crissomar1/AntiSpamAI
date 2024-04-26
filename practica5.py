import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from Main_ui import Ui_dialog
from DenseNet import *
from dibujar import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui



class Dialog(QMainWindow, Ui_dialog):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        #randominzar pesos
        self.inicializar()
        self.pushButton_Load.clicked.connect(self.load)
        self.pushButton_Plot.clicked.connect(self.plot)
        self.pushButton_Train.clicked.connect(self.train)

    def inicializar(self):
        self.modelo = DenseNet([2, 50,50, 1])
        self.doubleSpinBox_LR.setValue(1)
        self.epoch = 0
        self.label_Presicion.setText("")
        self.label_F1.setText("")
        self.label_MC.setText("")
        self.spinBox_Epochs.setValue(0)
        self.spinBox_Epochs.setRange(0, 1000)


    def train(self):
        self.modelo.fit(self.X, self.Y, self.spinBox_Epochs.value(), self.doubleSpinBox_LR.value())
        self.plot()
        self.presicion()

    def load(self):
        #cargar datos
        df = pd.read_csv('moons.csv')
        self.Y = np.asanyarray(df.iloc[:, 2]).T.reshape(1, -1)
        self.X = np.asanyarray(df.iloc[:, :2]).T




    def plot(self):
        #plot on frame
        plt.clf()
        MLP_binary_draw(self.X, self.Y, self.modelo)
        plt.title("Clasificación de datos")
        plt.grid()
        self.canvas = FigureCanvas(plt.gcf())
        self.canvas.setParent(self.frame)
        self.canvas.setGeometry(0, 0, 371, 311)
        self.canvas.show()

    def presicion(self):
        #calcular presicion en error
        Y_est = self.modelo.predict(self.X)
        Y_est = (Y_est > 0.5).astype(int)
        Y = self.Y
        #calcular matriz de confusion
        MC = self.confusion_matrix(Y, Y_est)
        #calcular F1
        F1 = 2 * MC[1, 1] / (2 * MC[1, 1] + MC[1, 0] + MC[0, 1])
        #calcular presicion
        presicion = np.sum(Y == Y_est) / Y.size
        self.label_Presicion.setText("Presicion: " + str(presicion))
        self.label_F1.setText("F1: " + str(F1))
        self.label_MC.setText("Matriz de confusion: \n" + str(MC))

    def confusion_matrix(self, Y, Y_est):
        #calcular matriz de confusion
        MC = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                MC[i, j] = np.sum((Y == i) & (Y_est == j))
        return MC

    def accept(self):
        pass

    def reject(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Dialog()
    form.show()
    sys.exit(app.exec_())