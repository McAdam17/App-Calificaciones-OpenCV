import sys
import os.path
import re  
import cv2
import numpy as np

def obtenerRespuestas(img):
    canny = cv2.Canny(img,20,150)
    kernel = np.ones((5,5),np.uint8)
    bordesDilatados = cv2.dilate(canny,kernel)
    contourns,_ = cv2.findContours(bordesDilatados,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    objetos=bordesDilatados.copy()
    cv2.drawContours(objetos,[max(contourns, key = cv2.contourArea)], -1, 255, thickness=-1)
    output = cv2.connectedComponentsWithStats(objetos,4,cv2.cv2.CV_32S)
    numObjs = output[0]
    labels = output[1]
    stats = output[2]
    maximaE = np.argmax(stats[:,4][1:])+1
    mascara = np.uint8(255*(maximaE==labels))
    contour,_ = cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contour[0]
    poligonoCH = cv2.convexHull(cnt)
    puntosDelCH = poligonoCH[:,0,:]
    m,n = mascara.shape
    ar = np.zeros((m,n))
    mascaraCH = np.uint8(cv2.fillConvexPoly(ar, puntosDelCH, 1))*255    
    vertices = cv2.goodFeaturesToTrack(mascaraCH,4,0.1,20)
    x=vertices[:,0,0]
    y=vertices[:,0,1]
    vertices = vertices[:,0,:]
    xo=np.sort(x)
    yo=np.sort(y)
    xn=np.zeros((1,4))
    yn=np.zeros((1,4))
    xn=(x==xo[2])*n+(x==xo[3])*n
    yn=(y==yo[2])*m+(y==yo[3])*m
    verticesN=np.zeros((4,2))
    verticesN[:,0]=xn
    verticesN[:,1]=yn
    vertices=np.int64(vertices)
    verticesN=np.int64(verticesN)
    h,_ = cv2.findHomography(vertices,verticesN)
    imgH = cv2.warpPerspective(img,h,(n,m))
    roi = imgH[:,np.uint64(0.25*n):np.uint64(0.84*n)]
    opciones = ['A','B','C','D','E','x']
    respuestas = []
    for i in range(0,26):
        pregunta = roi[np.uint64(i*(m/26)):np.uint64((i+1)*(m/26)),:]
        sumI = []
        for j in range(0,5):
            _,col = pregunta.shape
            sumI.append(np.sum(pregunta[:,np.uint64(j*(col/5)):np.uint64((j+1)*(col/5))]))
        vmin = np.ones((1,5))*np.min(sumI)
        minvalue = 0.17*col*m
        if np.linalg.norm(sumI-vmin)>minvalue:
            sumI.append(float('inf'))
        else:
            sumI.append(-1)
        respuestas.append(opciones[np.argmin(sumI)])
    return respuestas
def obtnerCalificacion(src):
    imagen = cv2.imread(src,0)
    respuestasFormato=np.array(obtenerRespuestas(imagen))
    respuestasCorrectas=np.array(['B','C','D','E','B','C','C','A','A','B','C','A','E','A','A','A','B','C','A','A','B','C','A','A','B','C'])
    calificacion=10*np.sum(respuestasCorrectas==respuestasFormato)/26
    return calificacion
src = str(sys.argv[1])
if os.path.isfile(src):
    cal = obtnerCalificacion(src)
    print('La calificcación es: ',cal)
elif os.path.isdir(src):
    suma = 0
    i = 0
    patron = re.compile('[a-zA-Z][a-zA-Z0-9_.-]*.jpg') 
    patronCarpeta = re.compile('[a-zA-Z][a-zA-Z0-9_.-]*/')
    if not (patronCarpeta.match(src)):
        src = src+'/'
    print('procesando ...')
    for filename in os.listdir(src):
        if(patron.match(filename)):
            suma = suma + float(obtnerCalificacion(src+filename))
            i = i + 1
    promedio = suma/i
    print('El promedio es: ',promedio) 
else:
    print('No se encontro el archivo indicado')
