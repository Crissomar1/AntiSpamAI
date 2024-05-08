import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from Main_ui import Ui_dialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui
import Email_Encoder

from tensorflow import keras


class Dialog(QMainWindow, Ui_dialog):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        #randominzar pesos
        self.inicializar()

    def inicializar(self):
        # cargar modelo
        self.model = keras.models.load_model('spam_model.keras', compile=True)
        if self.model:
            self.label_Error.setText('Modelo cargado')
        else:
            self.label_Error.setText('Error al cargar modelo')
        # read vocab
        self.vocab_dict = Email_Encoder.read_vocab()
        if self.vocab_dict:
            self.label_Error.setText('Vocabulario cargado')
        else:
            self.label_Error.setText('Error al cargar vocabulario')

    def on_pushButton_Load_clicked(self):
        #abrir archivo
        filename = QFileDialog.getOpenFileName(self, 'Open Mail File', '')
        if filename[0]:
            #encode email
            encoded_email = Email_Encoder.encode_email(filename[0], self.vocab_dict)
            #predict
            prediction = self.model.predict(np.array([encoded_email]))
            self.label_Label.setText("Spam" if prediction[0] > 0.5 else "No Spam")
            self.label_Error.setText('')


    def accept(self):
        pass

    def reject(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Dialog()
    form.show()
    sys.exit(app.exec_())