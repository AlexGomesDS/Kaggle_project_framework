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


def main():
	repo_path = '.'
	if len(sys.argv) < 2:
		print('Correct usage: python install.py repository_name [repository path]')
		return -1
	elif len(sys.argv) == 3:
		repo_path = sys.argv[2]
	
	repo_name = sys.argv[1]
	with urlopen(framework_repo_zip_url) as zipresp:
		with ZipFile(BytesIO(zipresp.read())) as zfile:
			zfile.extractall(members = [i for i in zfile.namelist() if re.match('Kaggle_project_framework-master/.*', i)])
		
	os.rename('Kaggle_project_framework-master', repo_name)
	
	with open(repo_name+'\\Readme.md','w') as readme_file:
		readme_file.write('# Repository created from https://github.com/AlexGomesDS/Kaggle_project_framework\n\n' + repo_name + '\n (Insert description here)')
	
	print('Succes, repository copied to: {}\nDon\'t forget to add it in github'.format(repo_path + '\\' + repo_name))
	return 0

if __name__ == '__main__':
  main()



