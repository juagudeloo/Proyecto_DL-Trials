all: all_process

all_process: %.pkl
	python3 Obtain_values.py
%.pkl: Train_model.py
	rm Images/Stokes_params/*
	python3 Train_model.py

train_model:
	python3 Train_model.py

obtain_values:
	python3 Obtain_values.py

create_scaler:
	python3 scaler_save.py
	rm *.npy

clean_impressed:
	rm *.npy

clean_images:
	rm Images/Stokes_params/*
	rm Images/Intensity/*

clean_images_cloud:
	git rm Images/Stokes_params/*
	git rm Images/Intensity/*