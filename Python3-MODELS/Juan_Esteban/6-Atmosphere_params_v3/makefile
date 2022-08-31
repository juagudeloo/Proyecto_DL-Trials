all: all_process

all_process: %.pkl Main.py
	rm Images/Stokes_params/*
	rm Images/Intensity/*
	python3 Main.py
%.pkl: scaler_save.py
	python3 scaler_save.py

main_code:
	rm Images/Stokes_params/*
	rm Images/Intensity/*
	python3 Main.py

create_scaler:
	python3 scaler_save.py

clean_impressed:
	rm *.npy

clean_images:
	rm Images/Stokes_params/*
	rm Images/Intensity/*
