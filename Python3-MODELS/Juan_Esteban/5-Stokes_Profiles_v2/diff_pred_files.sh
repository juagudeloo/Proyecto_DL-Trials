#!/bin/bash
t=100000
for i in $(seq 53000 10000 223000);
do
if ((i < t)); 
then 
echo "0$i" | python3 Main_code.py;
else 
echo "$i" | python3 Main_code.py;
fi
done
