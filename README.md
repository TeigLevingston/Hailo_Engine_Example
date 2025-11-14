This is the start of a project I am working on. I have a high-resolution camera, outputting 2560x1440, and I am testing the performance of the standard Hailo streaming detection approach, where the high-resolution stream is resized to match the model input of 640x640, against a motion detection algorithm that crops the high-resolution frame for sections where motion is detected to the model input size.

Additionally, since I don’t copy and paste code from Github or SO without understanding it, I wanted to document the process of creating and calling the Hailo8 inferencing engine. Hopefully, if anyone reads this, they'll find it helpful. Comments are welcome.

This is example code for my use so it isn't production ready. All of the functions are in a single file to make it a little easier to debug it and see what is going on as the model loads and answers the infer calls. There are also way too many comments but they are my notes for understanding the process.
