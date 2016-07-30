# For testing gpu vs. cpu speed of various networks
for OUTPUT in $(ls tests/*)
do
	th $OUTPUT
done