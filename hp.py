import numpy as np
import matplotlib.pyplot as plt
from manim import *

import math as mt


np.random.seed(42)
X_data = np.random.rand(100, 1) * 10  
Y_data = 3 * X_data + np.random.randn(100, 1) * 2  
X=X_data.reshape(-1)
Y=Y_data.reshape(-1)
points = list(zip(X, Y))
def costfun(x,y,slope,bias,n):
    error=(1/(2*n))*np.sum((y-(x*slope + bias))**2)
    dm=-1*np.mean(x*(y-(x*slope + bias)))
    dc=-1*np.mean(y-(x*slope + bias))
    return error,dm,dc

def LinearReg(x,y,iters):
    m,c=1,1 
    learning_rate=0.005
    ms=[]
    cs=[]
    es=[]
    for i in range(iters):
        err,dm,dc=costfun(x,y,m,c,y.shape[0])
        m-=(learning_rate)*dm
        c-=(learning_rate)*dc
        ms.append(m)
        cs.append(c)
        es.append(err)
        learning_rate-=(learning_rate)*0.005
    return ms,cs,es



class graph(Scene):

    def construct(self):    
        image = ImageMobject("123.png")
        text = Text("Data Dissection")
        text.scale(0.5)
        image.to_edge(UP)
        text.next_to(image, DOWN)
        text.shift(UP)
        image.scale(0.5)
        self.add(image,text)
        
        # Linear Regression call
        m,c,e=LinearReg(X_data,Y_data,20)
        # defines the axes and linear function
        axes = Axes(x_range=[-1, 10,1], y_range=[-6, 60,6], x_length=7, y_length=7)
        axes2 = Axes(x_range=[-8, 8,1], y_range=[-6, 64,6], x_length=7, y_length=7)
        func = axes.plot(lambda x: x, color=BLUE)
        error_fun = axes2.plot(lambda x: (x**2), color=BLUE)
        dot = Dot(radius=0.3)
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=RED) for x, y in points
        ])
        grph2=Group(axes2, error_fun,dot)
        grph2.to_corner(RIGHT)
        grph2.scale(0.5)
        grph=Group(axes, func,dots)
        grph.to_corner(LEFT)
        grph.scale(0.5)
        self.add(grph) 
        self.add(grph2) 
        for i in range(0,19,1):
            func2 = axes.plot(lambda x: m[i]*x + c[i]  , color=BLUE)
            self.play(dot.animate.move_to(axes2.c2p(mt.sqrt(e[i]),e[i])))
            self.play(Transform(func,func2))
        self.wait(1)
        

    