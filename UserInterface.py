import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from Translation_System import start_translation
import cv2
import tkinter as tk
from tkinter import Toplevel, Label
from PIL import Image, ImageTk


def show_info():
      # FUNCTION CREATES A POPUP WINDOW THAT DISPLAYS DETAILS ABOUT THE SYSTEM
    info_win = Toplevel(root)
    info_win.title("System Information")
    info_win.configure(bg="#1e1e1e")
    info_win.geometry("600x400")
    info_win.resizable(False, False)

    logo_img = Image.open("communication.png")  
    logo_img = logo_img.resize((120, 120), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)

    logo_label = Label(info_win, image=logo_photo, bg="#1e1e1e")
    logo_label.image = logo_photo
    logo_label.pack(pady=(30, 10))

    heading = Label(
        info_win,
        text="South African Sign Language Translator",
        font=("Helvetica", 20, "bold"),
        fg="#ffffff",
        bg="#1e1e1e"
    )
    heading.pack(pady=(0, 20))

    body_text = (
        "This system translates South African Sign Language (SASL) "
        "into text in real time using a Convolutional Neural Network (CNN) model.\n\n"
        "The built-in library currently recognizes 50 words.\n\n"
        "Examples include:  Aim   •   Angel   •   Adult"
    )

    body_label = Label(
        info_win,
        text=body_text,
        font=("Helvetica", 14),
        fg="#dcdcdc",
        bg="#1e1e1e",
        wraplength=520,
        justify="center"
    )
    body_label.pack(padx=30)

    #CLOSE BUTTON FUNCTIONS
    close_btn = tk.Button(
        info_win,
        text="Close",
        font=("Helvetica", 12, "bold"),
        bg="#2e8b57",
        fg="white",
        activebackground="#3cb371",
        activeforeground="white",
        relief="flat",
        command=info_win.destroy
    )
    close_btn.pack(pady=30)

def exit_application():
    root.quit()

def on_hover(event, widget):
    widget.config(bg="#444444")


def on_leave(event, widget):
    widget.config(bg="black")

# FUNCTIONS CALL START TRANSLATION METHOD
def begin_translation():
    for widget in menu_widgets:
        widget.pack_forget()
    start_translation()
    show_menu()

# LOADS MAIN WINDOW WITH BUTTON STYLING
root = tk.Tk()
root.title("Sign Language to Text Translator")
root.attributes("-fullscreen", True)

background_image = Image.open("C:/Users/Home/Desktop/Demo Project/backround.png")
background_image = background_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)
background_image = ImageTk.PhotoImage(background_image)

background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)


title_label = tk.Label(root, text="Sign Language to Text Translator", font=("Arial", 36),
                       fg="white", bg="#2e2e2e", relief="solid", bd=1)
title_label.pack(pady=40)

button_style = {"font": ("Arial", 24), "bg": "black", "fg": "white", "relief": "solid", "bd": 2, "height": 2}
button_width = 30

buttons = [
    ("Start Translation", begin_translation),
    ("Info", show_info),
    ("Exit", exit_application),
]

menu_widgets = []

for text, command in buttons:
    btn = tk.Button(root, text=text, width=button_width, **button_style, command=command)
    btn.pack(pady=10)
    btn.bind("<Enter>", lambda e, w=btn: on_hover(e, w))
    btn.bind("<Leave>", lambda e, w=btn: on_leave(e, w))
    menu_widgets.append(btn)

# SHOWS MENU AGAIN
def show_menu():
    for widget in menu_widgets:
        widget.pack(pady=10)

# ESCAPE KEY BUTTON EXIT HANDLER
def on_escape(event):
    show_menu()
    cap.release()
    cv2.destroyAllWindows()

root.bind("<Escape>", on_escape)

root.mainloop()
