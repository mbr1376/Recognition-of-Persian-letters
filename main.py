#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:33:30 2021

@author: mohamad
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:52:54 2020

@author: mohamad
"""


from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import cv2
import numpy as np
import random
import time    
from paintwidget import Window
from Classification import Mlp_clsiffication

class Mlp(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)

        loadUi("dising.ui",self)
        self.paint.clicked.connect(self.open_paint)  
        self.file.clicked.connect(self.open_file)
        self.Train.clicked.connect(self.Train_class)
        self.test.clicked.connect(self.test_file)
        self.consol.append("Mlp Classification.....")

    def Train_class(self):
        self.widget.canvas.axes.clear()
        self.file.setEnabled(True)
        self.paint.setEnabled(True)
        self.m=Mlp_clsiffication()
        training_accuracy,loss , acc=self.m.Train_classification()
        self.widget.canvas.axes.plot( training_accuracy,label=('train'))
        self.widget.canvas.axes.set_xlabel("Epoch")
        self.widget.canvas.axes.set_ylabel("Loss")
        self.widget.canvas.axes.legend()
        self.widget.canvas.draw()
        self.consol.append("end Train \n"+"Loss:"+str(loss) +"     Acc" +str(acc))
    def open_file(self):
        self.m=Mlp_clsiffication()
        self.m.read_file()
        self.consol.append("see classify directory")
        
        
        
        
        
    def open_paint(self):         
         self.ui=Window();
         self.ui.show()
         self.test.setEnabled(True)
         
         
    def test_file(self):
         self.m=Mlp_clsiffication()
         y=self.m.Test()  
         if(y==0):
             self.consol.append("peridict drow paint A")
         if(y==1):
             self.consol.append("peridict drow paint B")
         if(y==2):
             self.consol.append("peridict drow paint D")
         if(y==3):
             self.consol.append("peridict drow paint G")
if __name__=="__main__":
    
    app = QApplication([])
    window = Mlp()
    window.show()
    app.exec_()