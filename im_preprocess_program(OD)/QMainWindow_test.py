# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:34:30 2021

@author: heaven

code source : https://wikidocs.net/book/2165
QMainWindow(official document) : https://doc.qt.io/qt-5/qmainwindow.html
QStatusBar(official document) : https://doc.qt.io/qt-5/qstatusbar.html
QMenuBar(official document) : https://doc.qt.io/qt-5/qmenubar.html
"""
# 필요한 모듈 구성
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp
from PyQt5.QtGui import QIcon



class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 메뉴바
        exitAction = QAction(QIcon('exit.png'), 'Exit', self) # 동작(아이콘과 라벨 포함) 생성
        exitAction.setShortcut('Ctrl+Q') # 동작 단축키
        exitAction.setStatusTip('Exit application') # 동작 메뉴에 마우스 올릴시 상태팁(상태바) 설정
        exitAction.triggered.connect(qApp.quit) # 동작 선택시 QApplication quit에 연결되는 트리거

        self.statusBar()

        menubar = self.menuBar() # 메뉴바 생성
        menubar.setNativeMenuBar(False) 
        filemenu = menubar.addMenu('&File') # & 는 간단하게 단축키 설정(F앞이므로 Alt+F)
        filemenu.addAction(exitAction) # 위의 동작 설정을 File액션에 추가
        self.create_action(self.close, _file_menu, '&Exits')
        
        # 상태바
        self.statusBar().showMessage('Ready') # 상태바 메세지

        self.setWindowTitle('My First Application') # 타이틀바에 나오는 창 제목
        self.setGeometry(300, 300, 300, 200) # 창의 x,y위치와 너비,높이
        self.show() # 위젯을 스크린에 띄움


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())