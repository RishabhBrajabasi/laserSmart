import pdfplumber
import argparse
from flask import Flask, render_template, send_file, redirect, url_for
from flask_socketio import SocketIO
import io
from cv2 import aruco
import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", type=str, help="path to the .pdf presentation")
parser.add_argument("--resolution", type=int, default=96, help="resolution to render slides at")
parser.add_argument("--marker_size", type=int, default=96, help="size to render the aruco marker")
opt = parser.parse_args()
print(opt)

pdf = pdfplumber.open(opt.pdf)

def extract_page_objects(pdf, page_number):
    page_objects = pdf.pages[page_number].objects
    rects = []
    for pc in page_objects:
        if pc == "char" or pc == "rect":
            continue
        for po in page_objects[pc]:
            bbox = (int(po['x0']), int(po['top']), int(po['x1']), int(po['bottom']))
            rects.append(bbox)
    for po in pdf.pages[page_number].extract_words(keep_blank_chars=True):
        bbox = (int(po['x0']), int(po['top']), int(po['x1']), int(po['bottom']))
        rects.append(bbox)
    return rects

print("converting slides")
slides = [page.to_image(resolution=opt.resolution) for page in pdf.pages]
print("finished conversion")

# initialize flask
app = Flask(__name__)
# initialize socketio
socketio = SocketIO(app)

# aruco code dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# slide index
current_idx = 0

# presentation viewer
@app.route("/")
def present():
    return render_template("present.html")

# command for going to next slide
@app.route("/next")
def next_slide():
    socketio.emit("next")
    return ""

# command for going to previous slide
@app.route("/previous")
def previous_slide():
    socketio.emit("previous")
    return ""

# command for refreshing slide
@app.route("/refresh")
def refresh_slide():
    socketio.emit("refresh")
    return ""

@app.route("/select_slide_item/<float:x>/<float:y>")
def select_slide_item(x, y):
    global current_idx
    # get slide
    slide = slides[current_idx].annotated 
    # compute non-normalized pixel coordinates
    sw, sh = slide.size
    px = x * sw
    py = y * sh
    # get all objects on slide
    slide_items = extract_page_objects(pdf, current_idx)
    # find all objects that contain the point
    matches = []
    for si in slide_items:
        left, top, right, bottom = si
        if px >= left and px <= right and py >= top and py <= bottom:
            area = (right - left) * (bottom - top)
            matches.append((si, area))
    # select the "most specific" match
    match = sorted(matches, key=lambda x: x[1])[0][0] if len(matches) > 0 else None
    if match is None:
        return ""
    slides[current_idx].draw_rect(match, fill=(0, 0, 0, 0), stroke_width=5)
    # send refresh command
    socketio.emit("refresh")
    return ""

@app.route("/draw_line/<float:x1>/<float:y1>/<float:x2>/<float:y2>")
def draw_line(x1, y1, x2, y2):
    global current_idx
    # get slide
    slide = slides[current_idx].annotated 
    # compute non-normalized pixel coordinates
    sw, sh = slide.size
    px1 = x1 * sw
    py1 = y1 * sh
    px2 = x2 * sw
    py2 = y2 * sh
    # draw line on slide
    draw = ImageDraw.Draw(slide)
    draw.line([(px1, py1), (px2, py2)], width=5, fill=(255, 0, 0))
    # send refresh command
    socketio.emit("refresh")
    return ""

# slide renderer
@app.route("/view/<int:idx>")
def render_slide(idx):
    # bounds check
    if idx >= len(pdf.pages):
        return redirect(url_for("render_slide", idx=len(pdf.pages) - 1))
    # render the pdf to an image
    slide = slides[idx].annotated
    # generate the aruco codes
    marker_tl = aruco.drawMarker(aruco_dict, 25, opt.marker_size) # top left marker
    marker_tl = Image.fromarray(marker_tl)
    marker_tr = aruco.drawMarker(aruco_dict, 50, opt.marker_size) # top right marker
    marker_tr = Image.fromarray(marker_tr)
    marker_br = aruco.drawMarker(aruco_dict, 100, opt.marker_size) # bottom right marker
    marker_br = Image.fromarray(marker_br)
    marker_bl = aruco.drawMarker(aruco_dict, 75, opt.marker_size) # bottom left marker
    marker_bl = Image.fromarray(marker_bl)
    # add the aruco codes onto the slide
    sw, sh = slide.size
    slide.paste(marker_tl, (0, 0))
    slide.paste(marker_tr, (sw - opt.marker_size, 0))
    slide.paste(marker_br, (sw - opt.marker_size, sh - opt.marker_size))
    slide.paste(marker_bl, (0, sh - opt.marker_size))
    # write the slide to a virtual file buffer that flask can send
    output = io.BytesIO()
    slide.convert('RGBA').save(output, format='PNG')
    output.seek(0, 0)
    global current_idx
    current_idx = idx
    return send_file(output, mimetype='image/png', as_attachment=False)

if __name__ == "__main__":
    app.run()