compile-nb2script:
	jupyter nbconvert --to script identify_ptms.ipynb
	jupyter nbconvert --to script dataset_split.ipynb

compile-nb2pdf:
	jupyter nbconvert --to pdf identify_ptms.ipynb

identify-ptms:
	poetry run python -m identify_ptms

