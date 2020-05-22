import cv2
import PySimpleGUI as sg
window = sg.Window('Demo Application', [[sg.Image(filename='', key='image')]], location=(800,400))

cap = cv2.VideoCapture(0)

while True:
    event, values = window.Read(timeout=20, timeout_key='timeout')
    if event is None: break
    window.FindElement('image').Update(data=cv2.imencode('.png', cap.read()[1])[1].tobytes())
