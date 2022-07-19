# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 20:36:50 2022

@author: fairuzrizqi
"""

import cv2 as cv
import imutils as im
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

image = cv.imread('D:\data/images/Cars74.png')
image = im.resize(image, width=500)
cv.imshow("Gambar Asli", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gambar Grayscale", gray)

blur = cv.bilateralFilter(gray, 11, 17, 17)
cv.imshow("Bilateral Filter", blur)

edgeDet = cv.Canny(blur, 170, 200)
cv.imshow("Canny Result", edgeDet)

(cnts, _) = cv.findContours(edgeDet.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:30]

NumberPlateCnt = None
license_plate = None
x = None
y = None
w = None
h = None

count = 0
for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        NumberPlateCnt = approx

        # Menyimpan Hasil Crop Plat
        x, y, w, h = cv.boundingRect(c)
        new_img = gray[y:y + h, x:x + w]
        cv.imwrite('D:\data/images/' + str("hasil") + '.png', new_img)

        break
cv.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
cv.imshow("Plat Nomer Yang Terdeteksi", image)

Cropped_img = "D:\data\images\hasil.png"
cv.imshow("Plat Nomer", cv.imread(Cropped_img))

# konversi gambar plat
plat = pytesseract.image_to_string(Cropped_img)
print('Number is:' + plat)

cv.waitKey(0)



