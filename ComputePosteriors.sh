#!/bin/bash


echo "    Recomputing Gravitational Wave posteriors on f_PBH (may take ~ 1 hour...)"
echo " "

for NOBS in 1 80
do
    for PRIOR in J LF
	do
	    python src/tabulate_posteriors_O3.py -N_obs $NOBS -prior $PRIOR -M_PBH 0.5
	done
done

for NOBS in 1 24000
do
    for PRIOR in J LF
	do
	    python src/tabulate_posteriors_ET.py -N_obs $NOBS -prior $PRIOR
	done
done