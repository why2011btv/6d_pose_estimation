for i in `ls *.png`
do
mv $i `printf  %.12d ${i%.jpg}`.jpg
done
