# Object-recognition-using-Raspberry-Pi-with-PiCam
Using pretrained networks, a Raspberry pi with its cam is used to recognize objects.
Multithreading is used for a faster clasification and to avoid the bottle-neck in the I/O of images.
The code generate boxes randomly and chooses the ones with the highest Intersection Over Union, then the image is printed with the boxes.
