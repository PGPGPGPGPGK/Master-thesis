# Lane_Keeping_assist_on_RasPi
Lane keeping assist system with the help of open cv and implementation on raspberry pi 4 
Firstly with the opencv we apply functions like,
#Converting RGB image to HSV colorspace for detecting a specific color lane.
#Then we apply canny image masking to detect the edges in the image.
#As we just require the lanes to be detected and not the whole vision we specify polygon of our interest
#With average slope intercept we get a smooth lane
#We get an single line at center from these two lanes and compare it with our vertical height image reference to get the steering angle.
#The steering angle is stabilized to obtain maximum accuracy with two lane detection and one lane detection.
#The input angle is feed to servo motor which turns the steering wheel accordingly.
#We  caliberate our speed with change in deviation of angle, so for max steering angle the speed is minimum 
