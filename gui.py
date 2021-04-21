import tkinter as tk
import tkinter.font as tkFont
from PIL import Image

class App:
    def __init__(self, root):
        #setting title
        root.title("")
        #setting window size
        width=600
        height=500
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GRadio_989=tk.Radiobutton(root)
        GRadio_989["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_989["font"] = ft
        GRadio_989["fg"] = "#333333"
        GRadio_989["justify"] = "center"
        GRadio_989["text"] = "Region Filling"
        GRadio_989.place(x=420,y=260,width=86,height=49)
        GRadio_989["command"] = self.GRadio_989_command

        GRadio_27=tk.Radiobutton(root)
        GRadio_27["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_27["font"] = ft
        GRadio_27["fg"] = "#333333"
        GRadio_27["justify"] = "center"
        GRadio_27["text"] = "Convex Hull"
        GRadio_27.place(x=420,y=370,width=91,height=58)
        GRadio_27["command"] = self.GRadio_27_command

        GRadio_589=tk.Radiobutton(root)
        GRadio_589["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_589["font"] = ft
        GRadio_589["fg"] = "#333333"
        GRadio_589["justify"] = "center"
        GRadio_589["text"] = "Connected Component Extraction"
        GRadio_589.place(x=90,y=370,width=87,height=54)
        GRadio_589["command"] = self.GRadio_589_command

        GRadio_850=tk.Radiobutton(root)
        GRadio_850["bg"] = "#5fb878"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_850["font"] = ft
        GRadio_850["fg"] = "#333333"
        GRadio_850["justify"] = "center"
        GRadio_850["text"] = "Original Image"
        GRadio_850.place(x=240,y=160,width=115,height=56)
        GRadio_850["command"] = self.GRadio_850_command

        GRadio_602=tk.Radiobutton(root)
        GRadio_602["bg"] = "#ff4500"
        ft = tkFont.Font(family='Times',size=10)
        GRadio_602["font"] = ft
        GRadio_602["fg"] = "#333333"
        GRadio_602["justify"] = "left"
        GRadio_602["text"] = "Boundary Extraction"
        GRadio_602.place(x=90,y=260,width=82,height=52)
        GRadio_602["command"] = self.GRadio_602_command

        GMessage_312=tk.Message(root)
        ft = tkFont.Font(family='Times',size=28)
        GMessage_312["font"] = ft
        GMessage_312["fg"] = "#333333"
        GMessage_312["justify"] = "center"
        GMessage_312["text"] = "Image Processing"
        GMessage_312.place(x=170,y=40,width=270,height=48)

        GMessage_51=tk.Message(root)
        ft = tkFont.Font(family='Times',size=28)
        GMessage_51["font"] = ft
        GMessage_51["fg"] = "#333333"
        GMessage_51["justify"] = "center"
        GMessage_51["text"] = "Assignment"
        GMessage_51.place(x=260,y=90,width=80,height=25)

    def GRadio_989_command(self):
        a=Image.open("1.jpg")
        a.show()

    def GRadio_27_command(self):
        print("command")


    def GRadio_589_command(self):
        print("command")


    def GRadio_850_command(self):
        print("command")


    def GRadio_602_command(self):
        print("command")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
