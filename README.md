# Kaggle project framework

This repository is an attempt to create a python framework to standardize processes that are common to many Kaggle challenges.
The idea is that when starting a new kaggle challenge you extend this framework and customize it for the specific needs of that challenge by configuring each of it's steps, which will be executed sequentlially.

In other words this framework can be described as a pipeline specifically for Kaggle challenges.

The main goal of this framework was to on one side to avoid repeating code over all the repositories and on the other to simplify it by moving all the usual procedures (like reading, cleanning and exporting data) outside of the main code.

## Instructions
clone this repo and add the folder /src to sys.path, for example if you cloned to

	  C:\GIT\Kaggle_project_framework
	  
run:
	
	import sys
	sys.path.append('C:/GIT/Kaggle_project_framework/src')

		
Now you can extend the class AbstractPreprocessor from anywhere like this:

	from features.preprocessor.abstract_preprocessor import AbstractPreprocessor 
