# Artificial cell generation

This project attempts to emulate the [dataset comprised of white blood cells collected at the Hospital Clinic of Barcelona](https://www.sciencedirect.com/science/article/pii/S2352340920303681) using open source technologies.  
An important characteristic of project is that the dataset is emulated from scratch using classic approaches of generating data, meaning if one were to use this project, they would only need to install the required Python packages.  
This is an early proof of concept and this project is currently being worked at in my free time. Expect bugs and ugly code.

<p align="center">
  <img src="https://i.imgur.com/HjiRtJp.jpg" />
  <p align="center">example image of an neutrophil</p>
</p>

# About

Oftentimes artificial data generation is accomplished in a way that the original dataset is manipulated using different mathematical methods such as rotation, inversion, scaling, etc. Rarely does one attempt to generate data from scratch, especially an image data because it can be a convoluted undertaking and with the recent rise of generative neural networks, it could be seen as pointless.  
So, what exactly is the purpose of this repository? Firstly, a proof of concept, that emulating a complex image dataset using pure math and image processing techniques is indeed possible, no matter how crude this example is currently. I do not have a medical background, which is an another hurdle when differentiating between types of white blood cells. Secondly, it is fun. And lastly, it could help with training neural networks, as the code in this repository automatically outputs annotations in a Yolov5 format, which can then be easily converted into different formats. It took 15 minutes on my i5-10500H to generate 1000 images, which means that in half an hour one could have a 2000 image custom dataset with minimal human effort to train their neural network on or anything else.  
Bear in mind that this repo is still in its' early phases and was made with the mentality of results first, readability second since it was a project for a master thesis. I will try to make the effort to clean up messes I've made, but no promises. In the meantime, if you want to help, scroll down to the contribute page section.
# How to run

Install all the dependencies from requirements.txt file in the repository. I **highly** recommend usage of *virtual enviroments* to have a clean install. The Python version used in the project is 3.11.1, but newer Python version *should* be fine since this project mainly relies on numpy and OpenCV.  
After that, run the gen_blob.py script and that's it. Edit the script to your preferences or needs.

# Contributing

Contribution is welcome in any form, be it bug reports, pull requests or general knowledge and advices. If you'd like to help, but don't know how, take a peek at the issues page, I'll open issues for features I'd like to implement and refactor.


