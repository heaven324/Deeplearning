# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:34:30 2021

@author: heaven

code source : https://wikidocs.net/book/2165
QtWidgets(official document) : http://pyqt.sourceforge.net/Docs/PyQt5/QtWidgets.html#PyQt5-QtWidgets
QApplication(official document) : https://doc.qt.io/qt-5/qapplication.html
QToolTip(official document) : http://doc.qt.io/qt-5/qtooltip.html
"""
# 필요한 모듈 구성
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QCoreApplication


class MyApp(QWidget):

    
    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        # tool tip
        QToolTip.setFont(QFont('SansSerif', 10)) # 툴팁에 사용될 폰트와 사이즈 설정
        self.setToolTip('This is a <b>QWidget</b> widget') # 표시될 텍스트 입력
        
        # generate botten(나가기)
        btn = QPushButton('나가기', self) # 버튼 텍스트, 위젯
        btn.setToolTip('This is a <b>exit</b> widget')
        btn.move(50, 50)
        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.close()) # 이벤트 처리(이벤트루프 종료) issue 1
        #btn.clicked.connect(QtWidgets.QApplication.quit) # 이벤트 처리(이벤트루프 종료) issue 1
        #sys.exit(btn.clicked.connect()) # 이벤트 처리(프로그램 종료) issue 1
        
        self.setWindowTitle('My First Application') # 타이틀바에 나오는 창 제목
        self.setWindowIcon(QIcon('ic.png')) # 어플리케이션 아이콘 설정
        self.setGeometry(300, 300, 300, 200) # 창의 x,y위치와 너비,높이
        self.show() # 위젯을 스크린에 띄움


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())
   
   
"""
issue 1 : 버튼클릭으로 프로그램이 종료되게 하려면?
"""