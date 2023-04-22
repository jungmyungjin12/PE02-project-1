import sys
from PyQt5.QtWidgets import *
import iv
import reference
import transmission_measured as tm
import transmission_processed as tp
import matplotlib.pyplot as plt
from get_result import *
import plot

class MyApp_search(QDialog):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.show()

    def initUI(self):
        self.setWindowTitle('search image')
        self.move(300,300)
        self.resize(380,400)
        grid = QGridLayout()
        self.setLayout(grid)

        self.Wafer_label = QLabel('Wafer',self)
        grid.addWidget(self.Wafer_label, 0, 0)
        self.Wafer_box = QComboBox(self)
        self.Wafer_box.addItem('D07'), \
        self.Wafer_box.addItem('D08'), \
        self.Wafer_box.addItem('D23'), \
        self.Wafer_box.addItem('D24'), \
        grid.addWidget(self.Wafer_box, 0, 1)

        self.Row_label = QLabel('Row : ',self)
        grid.addWidget(self.Row_label, 1, 0)
        self.Row_box = QComboBox(self)
        self.Row_box.addItem('-4'), \
        self.Row_box.addItem('-3'), \
        self.Row_box.addItem('-2'), \
        self.Row_box.addItem('-1'), \
        self.Row_box.addItem('0'), \
        self.Row_box.addItem('1'), \
        self.Row_box.addItem('2'), \
        self.Row_box.addItem('3'), \
        grid.addWidget(self.Row_box, 1, 1)

        self.Col_label = QLabel('Column : ',self)
        grid.addWidget(self.Col_label, 2, 0)
        self.Col_box = QComboBox(self)
        self.Col_box.addItem('-4'), \
        self.Col_box.addItem('-3'), \
        self.Col_box.addItem('-2'), \
        self.Col_box.addItem('-1'), \
        self.Col_box.addItem('0'), \
        self.Col_box.addItem('1'), \
        self.Col_box.addItem('2'), \
        self.Col_box.addItem('3'), \
        grid.addWidget(self.Col_box, 2, 1)

        self.IV_Check = QCheckBox('IV_graph(fitting)',self)
        grid.addWidget(self.IV_Check, 3, 0)
        self.Ts1_Check = QCheckBox('Transmission spectra(measured)',self)
        grid.addWidget(self.Ts1_Check, 4, 0)
        self.Ts2_Check = QCheckBox('Transmission spectra(processed)',self)
        grid.addWidget(self.Ts2_Check, 5, 0)
        self.Rf_Check = QCheckBox('Reference fitting',self)
        grid.addWidget(self.Rf_Check, 6, 0)
        self.all_Check = QCheckBox('all',self)
        grid.addWidget(self.all_Check, 7, 0)


        self.btn1 = QPushButton('Show',self)
        grid.addWidget(self.btn1, 9, 0)
        self.btn1.clicked.connect(self.test2)

        self.btn2 = QPushButton('Check', self)
        grid.addWidget(self.btn2, 9, 1)
        self.btn2.clicked.connect(self.test)

        self.all_Check.stateChanged.connect(self.turnoff4)

        self.Wafer_check = QLabel('Wafer : ', self)
        grid.addWidget(self.Wafer_check, 8, 0)

        self.Row_check = QLabel('Row : ', self)
        grid.addWidget(self.Row_check, 8, 1)

        self.Col_check = QLabel('Column : ', self)
        grid.addWidget(self.Col_check, 8, 2)

    def test(self):
        x = str(self.Col_box.currentText())
        y = str(self.Row_box.currentText())
        z = str(self.Wafer_box.currentText())
        self.Wafer_check.setText('Wafer : ' + z)
        self.Row_check.setText('Row : ' + y)
        self.Col_check.setText('Column : ' + x)

    def turnoff4(self):
        if self.all_Check.isChecked():
            self.IV_Check.setEnabled(False)
            self.Ts2_Check.setEnabled(False)
            self.Ts1_Check.setEnabled(False)
            self.Rf_Check.setEnabled(False)
        else:
            self.IV_Check.setEnabled(True)
            self.Ts2_Check.setEnabled(True)
            self.Ts1_Check.setEnabled(True)
            self.Rf_Check.setEnabled(True)

    def test2(self):
        x = str(self.Col_box.currentText())
        y = str(self.Row_box.currentText())
        z = str(self.Wafer_box.currentText())
        a = [z, y, x]

        if self.all_Check.isChecked() == True:
            for i in range(0, len(all_LMZ)):
                if TestSiteInfo(all_LMZ[i], "Wafer") == a[0] and TestSiteInfo(all_LMZ[i], "DieRow") == a[1] and TestSiteInfo(all_LMZ[i], "DieColumn") == a[2]:
                    plot.plot(all_LMZ[i])
                    plt.suptitle('Analysis_{}_({},{})_{}_{}'.format(TestSiteInfo(all_LMZ[i], "Wafer"),
                                                                    TestSiteInfo(all_LMZ[i], "DieRow"),
                                                                    TestSiteInfo(all_LMZ[i], "DieColumn"),
                                                                    TestSiteInfo(all_LMZ[i], 'TestSite'),
                                                                    Date(all_LMZ[i])))
                    plt.show()
            # print(a, "all")
        if self.IV_Check.isChecked() == True:
            for i in range(0, len(all_LMZ)):
                if TestSiteInfo(all_LMZ[i], "Wafer") == a[0] and TestSiteInfo(all_LMZ[i], "DieRow") == a[1] and TestSiteInfo(all_LMZ[i], "DieColumn") == a[2]:
                    iv.iv(all_LMZ[i])
                    plt.show()
            # print(a, "IV_graph(fitting)")
        if self.Ts1_Check.isChecked() == True:
            for i in range(0, len(all_LMZ)):
                if TestSiteInfo(all_LMZ[i], "Wafer") == a[0] and TestSiteInfo(all_LMZ[i], "DieRow") == a[1] and TestSiteInfo(all_LMZ[i], "DieColumn") == a[2]:
                    tm.measured(all_LMZ[i])
                    plt.show()
            # print(a, "Transmission spectra(measured)")
        if self.Ts2_Check.isChecked() == True:
            for i in range(0, len(all_LMZ)):
                if TestSiteInfo(all_LMZ[i], "Wafer") == a[0] and TestSiteInfo(all_LMZ[i], "DieRow") == a[1] and TestSiteInfo(all_LMZ[i], "DieColumn") == a[2]:
                    tp.processed(all_LMZ[i])
                    plt.show()
            # print(a, "Transmission spectra(processed)")
        if self.Rf_Check.isChecked() == True:
            for i in range(0, len(all_LMZ)):
                if TestSiteInfo(all_LMZ[i], "Wafer") == a[0] and TestSiteInfo(all_LMZ[i], "DieRow") == a[1] and TestSiteInfo(all_LMZ[i], "DieColumn") == a[2]:
                    reference.reference(all_LMZ[i])
                    plt.show()
            # print(a, "Reference fitting")

