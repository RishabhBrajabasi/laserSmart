from pdf2image import convert_from_path, convert_from_bytes
import cv2
import glob
import img2pdf

slides = glob.glob('/home/rishabh/17728/Aruco_Tracker/slides/*.jpg')
slides.sort()

with open("output.pdf", "wb") as f:
    f.write(img2pdf.convert([i for i in slides if i.endswith(".jpg")]))


