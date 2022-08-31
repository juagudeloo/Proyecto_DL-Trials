all: all_process

all_process: %.pkl Main.py
	rm Images/Stokes_params/*
	rm Images/Intensity/*
	python3 Main.py
%.pkl: scaler_save.py
	python3 scaler_save.py

train_model:
	python3 Train_model.py

obtain_values:
	rm Images/Stokes_params/*
	rm Images/Intensity/*
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