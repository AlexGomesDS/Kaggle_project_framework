# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:44:41 2018

@author: ASSG
"""

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import re, os, sys

framework_repo_zip_url = 'https://github.com/AlexGomesDS/Kaggle_project_framework/archive/master.zip'
repo_path = '.'


def main():
	if len(sys.argv) < 2:
		print('Correct usage: python install.py repository_name [repository path]')
		return -1
	elif len(sys.argv) == 3:
		repo_path = sys.argv[2]
	repo_name = sys.argv[1]
	full_repo_path = repo_path + '/' + repo_name
	
	with urlopen(framework_repo_zip_url) as zipresp:
		with ZipFile(BytesIO(zipresp.read())) as zfile:
			files_to_extract = zfile.namelist()
			files_to_extract.remove('Kaggle_project_framework-master/install.py')
			zfile.extractall(path = repo_path, members = files_to_extract)
		
	os.rename(repo_path+'/Kaggle_project_framework-master', full_repo_path)
	
	with open(full_repo_path + '/Readme.md','w') as readme_file:
		readme_file.write('# Repository created from https://github.com/AlexGomesDS/Kaggle_project_framework\n\n' + repo_name + '\n (Insert description here)')
	
	print('Succes, repository copied to: {}\nDon\'t forget to add it in github'.format(repo_path + '\\' + repo_name))
	return 0

if __name__ == '__main__':
  main()



