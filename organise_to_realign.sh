main_dir="/e/MASTER"
subjects="Barlovic_Radojka_19480907 Gili_Attilio_19290519 Pilger_Valerie_19720713 Comte_Paulette_19280309 Lebedev_Alexey_19450908"

cd $main_dir
mkdir -p realigned_data

for subj in ${subjects} ; do
  echo "Processing ${subj}"
  mkdir realigned_data/${subj}
done
