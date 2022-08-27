all: main_code

main_code: %.pkl Main.py
	rm Images/Stokes_params/*
	rm Images/Intensity/*
	python3 Main.py
%.pkl: scaler_save.py
	python3 scaler_save.py

clean:
	rm *.pkl

clean_images:
	rm Images/Stokes_params/*
	rm Images/Intensity/*
