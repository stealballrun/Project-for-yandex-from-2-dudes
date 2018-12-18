#Interface
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLCDNumber, QLabel, QLineEdit
import PyQt5.QtGui as QtGui
process = 0
#class
class Example(QWidget):
         def __init__(self):
                  super().__init__()
                  self.initUI()



         def initUI(self):
                  self.setGeometry(400, 400, 400, 400)
                  self.setWindowTitle('Вход для настоящих мужзчин и леди')

                  self.button_1 = QPushButton('Ваши веса', self)
                  self.button_1.move(155, 150)
                  self.button_1.clicked.connect(self.getfileforrealmen1)

                  self.button_2 = QPushButton('Авторские веса', self)
                  self.button_2.move(150, 200)
                  self.button_2.clicked.connect(self.getfileforrealmen2)

                  self.button_3 = QPushButton('Обучить', self)
                  self.button_3.move(155, 250)
                  self.button_3.clicked.connect(self.getfileforrealmen3)



                  self.name_label = QLabel(self)
                  self.name_label.setText("Введите название файла: ")
                  self.name_label.move(10, 90)

                  self.name_input = QLineEdit(self)
                  self.name_input.move(200, 80)



         def getfileforrealmen1(self):
                  name = '1 ' + self.name_input.text()
                  filethatweneed = open('name.txt', 'w')
                  filethatweneed.write(name)
                  filethatweneed.close()

         
         def getfileforrealmen2(self):
                  name = '2 ' + self.name_input.text()
                  filethatweneed = open('name.txt', 'w')
                  filethatweneed.write(name)
                  filethatweneed.close()
         
         
         def getfileforrealmen3(self):
                  name = '3 ' + self.name_input.text()
                  filethatweneed = open('name.txt', 'w')
                  filethatweneed.write(name)
                  filethatweneed.close()




#start
if __name__ == '__main__':
         app = QApplication(sys.argv)
         ex = Example()
         ex.show()
         sys.exit(app.exec())