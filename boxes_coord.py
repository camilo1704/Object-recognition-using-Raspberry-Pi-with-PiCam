import numpy as np
import cv2
from keras.applications.mobilenet import preprocess_input, decode_predictions
import datetime

def boxes(N=100):
    """genera inicio y fin de box en coordenadas normalizadas, tendria que ajustarla as aspect ratio de la imagen para
    que tenga prob uniforme realmente """
    #  heigth,width,channels=im.shape
    x = np.random.uniform(0, 0.87, N)
    y = np.random.uniform(0, 0.87, N)
    box_lenght_y = np.zeros(N)
    box_lenght_x = np.zeros(N)
    counter = 0
    for i, j in zip(x, y):
        maxim = max(i, j)
        if maxim == i:
            aux = np.random.uniform(0.1, 1 - maxim)
            box_lenght_x[counter] = aux
            box_lenght_y[counter] = np.random.uniform(0.1 * aux, min(3 * aux, 1 - j))
        else:
            aux = np.random.uniform(0.1, 1 - maxim)
            box_lenght_y[counter] = aux
            box_lenght_x[counter] = np.random.uniform(0.1 * aux, min(3 * aux, 1 - i))
        counter += 1
    return np.stack((x, y), axis=-1), np.stack((x + box_lenght_x, y + box_lenght_y), axis=-1)


def box_to_imgsize(img_shape, ini, fin):
    """hay que pasarle img.shape, y las coordenadas de inicio y fin a cambiar, y la cantidad de puntos"""
    height = img_shape[1]
    width = img_shape[0]
    new_ini = [(int(i * height), int(j * width)) for i, j in zip(ini[:, 0], ini[:, 1])]
    new_fin = [(int(i * height), int(j * width)) for i, j in zip(fin[:, 0], fin[:, 1])]
    return new_ini, new_fin
def crop_boxes(image,N=100):
    """pasar imagen y número de boxes, genera N boxes en métrica unitaria, lo convierte a píxeles, corta la imagen """
    ini,fin=boxes(N)

    #print(image.shape)
    in_box_to_imgsize=datetime.datetime.now()
    ini,fin=box_to_imgsize(image.shape,ini,fin)
    fin_box_to_imgsize=datetime.datetime.now()
    print('box_to_img_size: ',(fin_box_to_imgsize-in_box_to_imgsize).total_seconds())
    images_crop=[]
    in_crop=datetime.datetime.now()
    for i in range(N):
       # print(image.shape)
        aux=image[ini[i][1]:fin[i][1], ini[i][0]:fin[i][0], :]

        #print(ini[i][0],fin[i][0])
        images_crop.append(aux) #creo q guarda & lo q es piola pq no ocupa memoria ?)
    fin_crop=datetime.datetime.now()
    print('crop_images: ',(fin_crop-in_crop).total_seconds())
    return images_crop,ini,fin

def eval_boxes(input_image,images,coord_ini,coord_fin,N,model,model_input_shape,threshold=0.5,dic=None):
   "paso las imágenes cortadas y evaluo prob, si tiene mayor que threshold guardo coordenadas, label y prob"
   good_crops=[]
   coord=[]
   probs=[]
   time_to_resize=0
   time_to_predict=0
   time_to_preprocess=0
   time_to_decodepreds=0
   time_appending=0
   for i in range(N):
        time_1=datetime.datetime.now()
        imag_rsz = cv2.resize(images[i], (model_input_shape, model_input_shape)) #no hace big copy
        time_2=datetime.datetime.now()
        to_pred = np.expand_dims(imag_rsz, axis=0)
        to_pred = preprocess_input(to_pred)
        time_3=datetime.datetime.now()
        preds = model.predict(to_pred)
        time_4=datetime.datetime.now()
        cod,name,max_prob = decode_predictions(preds, top=1)[0][0]
        time_5=datetime.datetime.now()
        if max_prob>threshold:
            good_crops.append(images[i])
            coord.append((coord_ini[i],coord_fin[i]))
            probs.append((name,max_prob))
        time_6=datetime.datetime.now()
        time_to_resize+=(time_2-time_1).total_seconds()
        time_to_preprocess+=(time_3-time_2).total_seconds()
        time_to_predict+=(time_4-time_3).total_seconds()
        time_to_decodepreds+=(time_5-time_4).total_seconds()
        time_appending+=(time_6-time_5).total_seconds()
  # print(len(probs),'shapeprob')
   print('time resizing: ',time_to_resize)
   print('time preprocesing: ',time_to_preprocess)
   print('time predicting: ', time_to_predict)
   print('time decoding preds: ', time_to_decodepreds)
   print('time appending', time_appending)
   draw_boxes(input_image,coord,probs)
   return good_crops,coord,probs
def draw_boxes(input_image,coord,probs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (255,0, 0)
    lineType = 2
    for i in range(len(probs)):
        cv2.putText(input_image, str(probs[i]),coord[i][0],font,fontScale,(255,0,0))
        cv2.rectangle(input_image, coord[i][0], coord[i][1], (255, 0, 0), 1).astype('int')
            #"""    font,
             #   fontScale,
              #  fontColor,
               # lineType)"""