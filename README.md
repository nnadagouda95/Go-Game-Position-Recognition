# Go-Game-Position-Recognition

The program runs on Python3.5.2
The following packages are needed - 
1. Numpy
2. Scipy
3. Scikit-learn
4. OpenCV 3.3
5. Matplotlib
6. Math
7. Itertools

The input required is on line 356. The variable 'input_files needs' to be set as 1 or 2.
Setting it as 1 uses images (frames) from the first video and 2 uses images (frames) from the second video. The rest of the program does not need any inputs.

It can be run as - python3 go_recognition.py

The frames were extracted from videos using this one line - ffmpeg -i video2.mp4 -r 0.2 output_%04d.png.
The name of the video is 'video2.mp4' and 0.2 lets us pick up one frame every 5 seconds (1/5 = 0.2). The parameter - 'output_%04d.png' saves those frames with the name output_%04d where 'd' is a counter. Thus they get saved as output_0002.png for example.

Running python3 go_recognition.py gives us both the image level accuracy and the overall accuracy for the video frames.