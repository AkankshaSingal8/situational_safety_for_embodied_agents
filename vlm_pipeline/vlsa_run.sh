cp requirements.txt requirements_psc.txt

# remove problematic lines
sed -i '/rerun-sdk/d' requirements_psc.txt
sed -i '/nvidia-/d' requirements_psc.txt
sed -i '/jax-cuda12/d' requirements_psc.txt
sed -i '/^-e /d' requirements_psc.txt

# fix typeguard
sed -i 's/typeguard==.*/typeguard==2.13.3/' requirements_psc.txt